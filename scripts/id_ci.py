#!/usr/bin/env python3
"""
Estimate class-wise intrinsic dimension with subsampling and confidence intervals.

Implements the Appendix C.2 subsampling protocol and evaluates multiple k values:
  - For each class, sample N datapoints without replacement
  - Estimate ID for k in {3, 10, 15, 20, 25} by default
  - Repeat R times
Adds per-class confidence intervals and plots them.
"""

import argparse
import csv
import json
import math
import os
import zlib

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from two_step_zoo.datasets.image import get_raw_image_tensors, image_tensors_to_dataset
from two_step_zoo.id_estimator.estimator import MLEIDEstimator


class _NullWriter:
    def write_checkpoint(self, *args, **kwargs):
        return None

    def write_scalar(self, *args, **kwargs):
        return None


DEFAULT_DATASETS = ["mnist", "fashion-mnist", "svhn", "cifar10", "cifar100"]


def _stable_hash(text):
    return zlib.adler32(text.encode("utf-8")) & 0xFFFFFFFF


def _estimate_id(dataset, *, k, batch_size, num_workers, pfix):
    cluster_cfg = {
        "num_clusters": 1,
        "id_estimates_save": None,
        "id_estimate_num_datapoints_per_class": -1,
        "max_k": k,
        "id_est_batch_size": batch_size,
        "n_id_est_workers": num_workers,
        "eval_every_k": False,
        "latent_k": k,
        "pfix": pfix,
    }
    estimator = MLEIDEstimator(cluster_cfg, writer=_NullWriter())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return float(estimator.estimate_id(loader))


def _mean_ci(values, confidence):
    if len(values) == 0:
        return 0.0, 0.0, 0.0

    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, mean, mean

    std = float(np.std(values, ddof=1))
    sem = std / math.sqrt(len(values))

    t_crit = 1.96
    if confidence < 1.0 and confidence > 0.0:
        try:
            from scipy import stats  # type: ignore
            t_crit = float(stats.t.ppf((1 + confidence) / 2.0, df=len(values) - 1))
        except Exception:
            t_crit = 1.96

    half_width = t_crit * sem
    return mean, mean - half_width, mean + half_width


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _plot_dataset(dataset, stats, *, ks, confidence, output_dir, class_names):
    class_ids = sorted(stats.keys())
    if not class_ids:
        return

    class_score = {}
    for class_id in class_ids:
        means = [stats[class_id]["per_k"][k]["mean"] for k in ks]
        class_score[class_id] = float(np.mean(means))

    class_ids = sorted(class_ids, key=lambda cid: class_score[cid])
    if len(class_ids) > 10:
        class_ids = class_ids[:5] + class_ids[-5:]

    num_classes = len(class_ids)
    fig, ax = plt.subplots(figsize=(10, 6))

    base_x = np.arange(num_classes)
    width = 0.8 / max(len(ks), 1)

    for idx, k in enumerate(ks):
        means = [stats[c]["per_k"][k]["mean"] for c in class_ids]
        ci_low = [stats[c]["per_k"][k]["ci_low"] for c in class_ids]
        ci_high = [stats[c]["per_k"][k]["ci_high"] for c in class_ids]
        yerr = [
            np.array(means) - np.array(ci_low),
            np.array(ci_high) - np.array(means),
        ]

        offset = (idx - (len(ks) - 1) / 2.0) * width
        ax.bar(base_x + offset, means, width=width, yerr=yerr, capsize=2, label=f"k={k}")

    ax.set_title(f"{dataset} class ID with {int(confidence * 100)}% CI")
    ax.set_xlabel("Class")
    ax.set_ylabel("ID estimate")
    ax.legend(ncol=min(len(ks), 5), fontsize=8, frameon=False)

    ax.set_xticks(base_x)
    labels = [class_names[cid] if cid < len(class_names) else str(cid) for cid in class_ids]
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelrotation=45)

    fig.tight_layout()
    fig_path = os.path.join(output_dir, f"{dataset}_class_id_ci.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Class-wise ID estimates with confidence intervals")
    parser.add_argument("--data-root", type=str, default="data/")
    parser.add_argument("--output-dir", type=str, default="analysis/id_ci")
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--subsample", type=int, default=1000)
    parser.add_argument("--subsample-cifar100", type=int, default=300)
    parser.add_argument("--ks", type=str, default="3,10,15,20,25")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--no-pfix", dest="pfix", action="store_false")
    parser.set_defaults(pfix=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the current ID estimator implementation.")

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    ks = [int(k.strip()) for k in args.ks.split(",") if k.strip()]
    if not ks:
        ks = [args.k]
    _ensure_dir(args.output_dir)

    all_results = {}

    for dataset in datasets:
        print(f"Loading dataset: {dataset}")
        try:
            images, labels, class_names = get_raw_image_tensors(
                dataset_name=dataset,
                train=True,
                data_root=args.data_root,
                class_ind=-1,
                return_class_names=True
            )
        except ValueError as exc:
            print(f"Skipping unsupported dataset {dataset}: {exc}")
            continue

        labels = labels.to(torch.long)
        if dataset == "svhn" and labels.max().item() == 10:
            # SVHN uses label 10 for digit 0 in torchvision
            labels = labels.clone()
            labels[labels == 10] = 0

        num_classes = int(labels.max().item()) + 1
        class_indices = [
            torch.where(labels == c)[0].cpu().numpy()
            for c in range(num_classes)
        ]

        dataset_results = {}
        for class_id in range(num_classes):
            indices = class_indices[class_id]
            subsample_size = args.subsample_cifar100 if dataset == "cifar100" else args.subsample
            estimates_by_k = {k: [] for k in ks}
            for rep in range(args.repeats):
                seed = args.seed + _stable_hash(dataset) + class_id * 1000 + rep
                rng = np.random.default_rng(seed)
                chosen = rng.choice(indices, size=subsample_size, replace=False)

                subset_images = images[chosen]
                subset_labels = labels[chosen]
                subset_dataset = image_tensors_to_dataset(
                    dataset_name=dataset,
                    dataset_role="train",
                    images=subset_images,
                    labels=subset_labels,
                    transforms=None
                )

                for k in ks:
                    estimate = _estimate_id(
                        subset_dataset,
                        k=k,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        pfix=args.pfix
                    )
                    estimates_by_k[k].append(estimate)
                    print(f"{dataset} class {class_id} k {k} rep {rep}: {estimate:.3f}")

            per_k_stats = {}
            for k in ks:
                mean, ci_low, ci_high = _mean_ci(estimates_by_k[k], args.confidence)
                per_k_stats[k] = {
                    "n": len(estimates_by_k[k]),
                    "mean": mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "samples": estimates_by_k[k],
                }

            dataset_results[class_id] = {
                "per_k": per_k_stats,
            }

        all_results[dataset] = {
            "ks": ks,
            "repeats": args.repeats,
            "subsample": args.subsample,
            "subsample_cifar100": args.subsample_cifar100,
            "confidence": args.confidence,
            "classes": dataset_results,
        }

        dataset_out_dir = os.path.join(args.output_dir, dataset)
        _ensure_dir(dataset_out_dir)

        json_path = os.path.join(dataset_out_dir, "class_id_ci.json")
        with open(json_path, "w") as f:
            json.dump(all_results[dataset], f, indent=2, sort_keys=True)

        csv_path = os.path.join(dataset_out_dir, "class_id_ci.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class_id", "k", "n", "mean", "ci_low", "ci_high"])
            for class_id, stats in sorted(dataset_results.items()):
                for k in ks:
                    k_stats = stats["per_k"][k]
                    writer.writerow([
                        class_id,
                        k,
                        k_stats["n"],
                        k_stats["mean"],
                        k_stats["ci_low"],
                        k_stats["ci_high"],
                    ])

        _plot_dataset(
            dataset,
            dataset_results,
            ks=ks,
            confidence=args.confidence,
            output_dir=dataset_out_dir,
            class_names=class_names,
        )

    summary_path = os.path.join(args.output_dir, "all_datasets_class_id_ci.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
