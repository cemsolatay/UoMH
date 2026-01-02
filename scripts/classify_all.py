#!/usr/bin/env python3
"""
Run supervised classification across datasets with optional ID-weighted loss.

Uses two_step_zoo.datasets for data access and class names.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from classification_models import ResNet18, ResNet34, VGG
from classification_utils import deterministic_shuffle, progress_bar
from two_step_zoo.datasets.image import get_raw_image_tensors
from two_step_zoo.id_estimator.estimator import MLEIDEstimator


DATASETS = ["mnist", "fashion-mnist", "svhn", "cifar10", "cifar100"]

DATASET_MEAN_STD = {
    "mnist": ([0.1307], [0.3081]),
    "fashion-mnist": ([0.2860], [0.3530]),
    "svhn": ([0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]),
    "cifar10": ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    "cifar100": ([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2761]),
}

DATASET_EPOCHS = {
    "mnist": 60,
    "fashion-mnist": 60,
    "svhn": 120,
    "cifar10": 150,
    "cifar100": 200,
}

DATASET_LR_MILESTONES = {
    "mnist": [30, 45, 55],
    "fashion-mnist": [30, 45, 55],
    "svhn": [60, 90, 105],
    "cifar10": [75, 110, 130],
    "cifar100": [60, 120, 160],
}

DATASET_VAL_FRACTION = {
    "cifar100": 0.001,
}


class ClassifyDataset(Dataset):
    def __init__(self, images, labels, mean, std, replicate_channels=False, resize_to=None):
        self.images = images
        self.labels = labels.long()
        self.replicate_channels = replicate_channels
        self.resize_to = resize_to

        mean = torch.tensor(mean, dtype=torch.float32)
        std = torch.tensor(std, dtype=torch.float32)
        if replicate_channels and mean.numel() == 1:
            mean = mean.repeat(3)
            std = std.repeat(3)

        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx].to(torch.float32) / 255.0
        if self.replicate_channels and img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        if self.resize_to is not None:
            if img.shape[1] != self.resize_to or img.shape[2] != self.resize_to:
                img = F.interpolate(
                    img.unsqueeze(0),
                    size=(self.resize_to, self.resize_to),
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0)

        img = (img - self.mean) / self.std
        return img, self.labels[idx], idx


class IDDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images.to(torch.float32)
        self.labels = labels.long()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], idx


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model(model_name, num_classes):
    if model_name == "res18":
        return ResNet18(num_classes=num_classes)
    if model_name == "res34":
        return ResNet34(num_classes=num_classes)
    if model_name == "vgg":
        return VGG("VGG19", num_classes=num_classes)
    raise ValueError(f"Unknown model: {model_name}")


def build_loaders(dataset, data_root, batch_size, val_fraction):
    images, labels, class_names = get_raw_image_tensors(
        dataset_name=dataset,
        train=True,
        data_root=data_root,
        class_ind=-1,
        return_class_names=True,
    )
    test_images, test_labels, _ = get_raw_image_tensors(
        dataset_name=dataset,
        train=False,
        data_root=data_root,
        class_ind=-1,
        return_class_names=True,
    )

    labels = labels.to(torch.long)
    test_labels = test_labels.to(torch.long)
    if dataset == "svhn" and labels.max().item() == 10:
        labels = labels.clone()
        labels[labels == 10] = 0
    if dataset == "svhn" and test_labels.max().item() == 10:
        test_labels = test_labels.clone()
        test_labels[test_labels == 10] = 0

    perm = deterministic_shuffle(np.arange(labels.shape[0]))
    val_size = int(val_fraction * labels.shape[0])
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    train_images = images[train_idx]
    train_labels = labels[train_idx]
    val_images = images[val_idx]
    val_labels = labels[val_idx]

    mean, std = DATASET_MEAN_STD[dataset]
    replicate_channels = dataset in ["mnist", "fashion-mnist"]
    resize_to = 32 if dataset in ["mnist", "fashion-mnist"] else None

    train_dataset = ClassifyDataset(
        train_images, train_labels, mean, std,
        replicate_channels=replicate_channels, resize_to=resize_to
    )
    val_dataset = ClassifyDataset(
        val_images, val_labels, mean, std,
        replicate_channels=replicate_channels, resize_to=resize_to
    )
    test_dataset = ClassifyDataset(
        test_images, test_labels, mean, std,
        replicate_channels=replicate_channels, resize_to=resize_to
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, class_names, train_images, train_labels


def estimate_class_ids(train_images, train_labels, num_classes, batch_size, num_workers, max_samples=None):
    cluster_cfg = {
        "num_clusters": 1,
        "id_estimates_save": None,
        "id_estimate_num_datapoints_per_class": -1,
        "max_k": 10,
        "id_est_batch_size": batch_size,
        "n_id_est_workers": num_workers,
        "eval_every_k": False,
        "latent_k": 10,
        "pfix": True,
    }
    estimator = MLEIDEstimator(cluster_cfg, writer=_NullWriter())
    rng = np.random.default_rng(0)

    ids = []
    for class_id in range(num_classes):
        class_idx = torch.where(train_labels == class_id)[0].cpu().numpy()
        if max_samples is not None and len(class_idx) > max_samples:
            class_idx = rng.choice(class_idx, size=max_samples, replace=False)

        class_images = train_images[class_idx]
        class_labels = train_labels[class_idx]
        dataset = IDDataset(class_images, class_labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        id_est = estimator.estimate_id(loader)
        ids.append(float(id_est))

    return ids


class _NullWriter:
    def write_checkpoint(self, *args, **kwargs):
        return None

    def write_scalar(self, *args, **kwargs):
        return None


def build_loss(weights, device):
    if weights is None:
        return nn.CrossEntropyLoss()
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weights)


def update_lr(optimizer, new_lr):
    for group in optimizer.param_groups:
        group["lr"] = new_lr


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss_total += criterion(outputs, targets).item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / max(total, 1)
    return acc, loss_total / max(len(loader), 1)

def evaluate_per_class(model, loader, device, num_classes):
    model.eval()
    correct = torch.zeros(num_classes, dtype=torch.long)
    total = torch.zeros(num_classes, dtype=torch.long)
    overall_correct = 0
    overall_total = 0

    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1)
            overall_total += targets.size(0)
            overall_correct += (preds == targets).sum().item()
            for c in range(num_classes):
                mask = targets == c
                total[c] += mask.sum().item()
                if mask.any():
                    correct[c] += (preds[mask] == c).sum().item()

    accs = []
    totals = []
    for c in range(num_classes):
        tot = total[c].item()
        totals.append(tot)
        accs.append(100.0 * correct[c].item() / tot if tot > 0 else 0.0)
    overall_acc = 100.0 * overall_correct / max(overall_total, 1)
    return accs, totals, overall_acc


def train_one_run(model, train_loader, val_loader, test_loader, device, epochs, lr, milestones, criterion, num_classes):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    best_val = -1.0
    best_test = -1.0
    best_epoch = -1
    best_per_class = None
    best_per_class_counts = None

    for epoch in tqdm(range(epochs), desc="Epochs", leave=True):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets, _) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(train_loader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)" % (
                    running_loss / (batch_idx + 1),
                    100.0 * correct / max(total, 1),
                    correct,
                    total,
                )
            )

        if epoch in milestones:
            update_lr(optimizer, lr * (0.2 ** (milestones.index(epoch) + 1)))

        val_acc, _ = evaluate(model, val_loader, device)

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_per_class, best_per_class_counts, best_test = evaluate_per_class(
                model, test_loader, device, num_classes
            )

    return {
        "best_val_acc": best_val,
        "test_acc_at_best_val": best_test,
        "best_epoch": best_epoch,
        "per_class_test_acc_at_best_val": best_per_class,
        "per_class_test_counts": best_per_class_counts,
    }


def main():
    parser = argparse.ArgumentParser(description="Run classification models with ID-weighted loss for multiple datasets")
    parser.add_argument("--model", type=str, required=True, choices=["res18", "res34", "vgg"])
    parser.add_argument("--data-root", type=str, default="data/")
    parser.add_argument("--results-dir", type=str, default="runs/classification")
    parser.add_argument("--datasets", type=str, default=",".join(DATASETS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss-weighting", type=str, default="both", choices=["none", "id", "both"])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ID estimation.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        cudnn.benchmark = True

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    os.makedirs(args.results_dir, exist_ok=True)

    for dataset in datasets:
        if dataset not in DATASET_MEAN_STD:
            print(f"Skipping unknown dataset {dataset}")
            continue

        val_fraction = DATASET_VAL_FRACTION.get(dataset, 0.01)
        train_loader, val_loader, test_loader, class_names, train_images, train_labels = build_loaders(
            dataset=dataset,
            data_root=args.data_root,
            batch_size=args.batch_size,
            val_fraction=val_fraction
        )
        num_classes = int(train_labels.max().item()) + 1

        id_estimates = estimate_class_ids(
            train_images=train_images,
            train_labels=train_labels,
            num_classes=num_classes,
            batch_size=args.batch_size,
            num_workers=0,
            max_samples=None,
        )
        id_sum = sum(id_estimates) if sum(id_estimates) > 0 else 1.0
        class_weights = [w / id_sum for w in id_estimates]
        class_weights = [w * num_classes for w in class_weights]

        dataset_dir = os.path.join(args.results_dir, dataset, args.model)
        os.makedirs(dataset_dir, exist_ok=True)

        with open(os.path.join(dataset_dir, "class_id_weights.json"), "w") as f:
            json.dump({
                "class_names": class_names,
                "id_estimates": id_estimates,
                "weights": class_weights,
            }, f, indent=2, sort_keys=True)

        weight_modes = []
        if args.loss_weighting in ["none", "both"]:
            weight_modes.append("none")
        if args.loss_weighting in ["id", "both"]:
            weight_modes.append("id")

        for weight_mode in weight_modes:
            run_results = []
            for rep in range(args.repeats):
                print(f"Dataset {dataset} | model {args.model} | weighting {weight_mode} | rep {rep}")
                set_seed(args.seed + rep)

                model = get_model(args.model, num_classes).to(device)
                if device == "cuda":
                    model = torch.nn.DataParallel(model)

                weights = class_weights if weight_mode == "id" else None
                criterion = build_loss(weights, device)

                results = train_one_run(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=device,
                    epochs=DATASET_EPOCHS[dataset],
                    lr=0.1,
                    milestones=DATASET_LR_MILESTONES[dataset],
                    criterion=criterion,
                    num_classes=num_classes,
                )
                run_results.append(results)

            summary = {
                "dataset": dataset,
                "model": args.model,
                "weighting": weight_mode,
                "repeats": args.repeats,
                "results": run_results,
            }
            with open(os.path.join(dataset_dir, f"summary_{weight_mode}.json"), "w") as f:
                json.dump(summary, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
