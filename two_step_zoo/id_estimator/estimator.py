from .mle import mle_inverse_singlek
from .geomle import geomle
from .twonn import twonn
from .utils import dotdict
import math
import torch
import pdb

class IDEstimator:
    def __init__(self, cluster_cfg, writer):
        self.cluster_cfg = cluster_cfg
        self.id_estimates_save_name = cluster_cfg["id_estimates_save"]

        self.writer = writer

        self.estimates = [[] for _ in range(cluster_cfg["num_clusters"])]

    @property
    def has_estimates(self):
        for estimate in self.estimates:
            if isinstance(estimate, (list, tuple)):
                if len(estimate) <= self.cluster_cfg["latent_k"]:
                    return False
            else:
                if estimate is None:
                    return False
        return True
        
    def get_id_estimates(self, dataloaders):
        "Gives MLE estimate from dataloader"
        
        self.load_id_estimates()
        
        if not self.has_estimates:
            print("Could not load required ID estimates, running estimator")
            for idx,dataloader in enumerate(dataloaders):
                self.estimates[idx] = self.estimate_id(dataloader["train"])
       
        self.save_id_estimates()

        extracted = []
        for estimate in self.estimates:
            if isinstance(estimate, (list, tuple)):
                extracted.append(estimate[self.cluster_cfg["latent_k"]])
            else:
                extracted.append(estimate)
        return [math.ceil(value) for value in extracted]
    
    @property 
    def should_save_id_estimates(self): return self.id_estimates_save_name is not None

    def save_id_estimates(self):
        if self.should_save_id_estimates:
            print(f"Saving ID estimates to {self.id_estimates_save_name}")
            self.writer.write_checkpoint(self.id_estimates_save_name, self.estimates, absolute_path=True)

        for cidx,estimate in enumerate(self.estimates):
            if isinstance(estimate, (list, tuple)):
                value = estimate[self.cluster_cfg["latent_k"]]
            else:
                value = estimate
            self.writer.write_scalar("id_estimate_values", value, cidx)

    def load_id_estimates(self):
        try:
            self.estimates = self.writer.load_checkpoint(self.id_estimates_save_name, "cpu", absolute_path=True)
            print(f"Loaded ID estimates from {self.id_estimates_save_name}")
        except:
            print("Could not load ID estimates from checkpoint")

    def estimate_id(self, dataloader, dataset=False):
        "Gives ID estimate from dataloader"
        pass


class MLEIDEstimator(IDEstimator):
    def estimate_id(self, dataloader, dataset=False):
        
        if not dataset and self.cluster_cfg["id_estimate_num_datapoints_per_class"] != -1 and len(dataloader.dataset) > self.cluster_cfg["id_estimate_num_datapoints_per_class"]:
            _, inv_mle_dim,_ = mle_inverse_singlek(torch.utils.data.Subset(dataloader.dataset, [i for i in range(self.cluster_cfg["id_estimate_num_datapoints_per_class"])]) \
                if not dataset else dataloader, k1=self.cluster_cfg["max_k"], args=dotdict(self.cluster_cfg))
        else:
            _, inv_mle_dim,_ = mle_inverse_singlek(dataloader.dataset if not dataset else dataloader, k1=self.cluster_cfg["max_k"], args=dotdict(self.cluster_cfg))
        return inv_mle_dim


class GeoMLEIDEstimator(IDEstimator):
    def estimate_id(self, dataloader, dataset=False):
        full_dataset = dataloader.dataset if not dataset else dataloader
        args = dotdict({
            "bsize": self.cluster_cfg.get("geomle_batch_size", self.cluster_cfg["id_est_batch_size"]),
            "n_workers": self.cluster_cfg.get("geomle_num_workers", self.cluster_cfg["n_id_est_workers"]),
            "inv_mle": self.cluster_cfg.get("geomle_inv_mle", False),
        })
        k1 = self.cluster_cfg.get("geomle_k1", 20)
        k2 = self.cluster_cfg.get("geomle_k2", 55)
        nb_iter1 = self.cluster_cfg.get("geomle_bootstrap_subsets", 20)
        nb_iter2 = self.cluster_cfg.get("geomle_bootstrap_subsets_inner", 1)
        degree = self.cluster_cfg.get("geomle_degree", (1, 2))
        alpha = self.cluster_cfg.get("geomle_alpha", 5e-3)

        estimates = geomle(
            full_dataset,
            k1=k1,
            k2=k2,
            nb_iter1=nb_iter1,
            nb_iter2=nb_iter2,
            degree=degree,
            alpha=alpha,
            args=args
        )
        return float(estimates.mean())


class TwoNNIDEstimator(IDEstimator):
    def estimate_id(self, dataloader, dataset=False):
        full_dataset = dataloader.dataset if not dataset else dataloader
        args = dotdict({
            "bsize": self.cluster_cfg.get("twonn_batch_size", self.cluster_cfg["id_est_batch_size"]),
            "n_workers": self.cluster_cfg.get("twonn_num_workers", self.cluster_cfg["n_id_est_workers"]),
            "anchor_ratio": self.cluster_cfg.get("twonn_anchor_ratio", 0),
            "anchor_samples": self.cluster_cfg.get("twonn_anchor_samples", -1),
        })
        return float(twonn(full_dataset, args=args))
