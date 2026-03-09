"""
baselines/common.py
-------------------
Shared utilities used by all baseline strategies:
  - IID-level → Dirichlet-alpha mapping
  - ALIE malicious-network hook
  - Re-exported metric aggregation helpers
  - CSV writing helpers
  - Timing utilities
"""
from __future__ import annotations

import os
import time
import json
from enum import Enum
from typing import Optional, Union, Dict, List

import numpy as np
import torch

from flwr.common import (
    Context,
    FitRes,
    Scalar,
    MetricsAggregationFn,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy

from byebye_badclients.task import mean_and_sigma_flat, unflatten_parameters
from byebye_badclients.result_processing import list_to_csv, dict_to_csv


# ---------------------------------------------------------------------------
# IID-level mapping
# ---------------------------------------------------------------------------

_IID_LEVEL_MAP: dict[float, tuple[bool, float]] = {
    1.0: (False, 0.0),   # Fully IID – uniform random split
    0.7: (True,  1.0),  # Slightly Non-IID
    0.5: (True,  0.7),   # Moderately Non-IID
    0.3: (True,  0.5),   # Strongly Non-IID
    0.1: (True,  0.1),   # Extremely Non-IID
}


def iid_level_to_params(iid_level: float) -> tuple[bool, float]:
    """Convert an iid_level float to (non_iid, dirichlet_alpha).

    The closest supported level is selected via argmin on the key list.

    Returns
    -------
    (non_iid, dirichlet_alpha) : tuple[bool, float]
    """
    supported = sorted(_IID_LEVEL_MAP.keys())
    closest = min(supported, key=lambda k: abs(k - iid_level))
    return _IID_LEVEL_MAP[closest]


# ---------------------------------------------------------------------------
# ALIE malicious-network hook
# ---------------------------------------------------------------------------

def alie_malicious_network_hook(results: list[tuple[ClientProxy, FitRes]]) -> None:
    """Apply ALIE attack bias to any client whose attack_pattern == 'alie'.

    Modifies `fit_res.parameters` in-place for ALIE clients.
    """
    if not results:
        return

    alie_parameters = []
    alie_entries: list[tuple[ClientProxy, FitRes]] = []

    for client, fit_res in results:
        if fit_res.metrics.get("attack_pattern") == "alie":
            alie_parameters.append(fit_res.parameters)
            alie_entries.append((client, fit_res))

    if not alie_entries:
        return

    mu, sigma = mean_and_sigma_flat(parameters_list=alie_parameters)
    sigma = torch.tensor(sigma) * 1.5
    mu = mu - sigma

    for _, alie_res in alie_entries:
        g = 0.9 * mu + 0.1 * torch.randn_like(mu)
        alie_res.parameters = unflatten_parameters(g, alie_parameters[0])


# ---------------------------------------------------------------------------
# Metrics aggregation factories
# ---------------------------------------------------------------------------

def _collect_fit_results(
    results: dict,
    tracker: list[int],
    metrics_dict: dict[str, float],
    num_rounds: int,
    out_dir: str,
) -> None:
    tracker[0] += 1
    for k, v in metrics_dict.items():
        results.setdefault(k, []).append(v)
    if tracker[0] >= num_rounds:
        os.makedirs(out_dir, exist_ok=True)
        for name, vals in results.items():
            filepath = os.path.join(out_dir, f"{name}.csv")
            list_to_csv(l=vals, filepath=filepath, column_name=name)


def _collect_eval_results(
    results: dict,
    tracker: list[int],
    metrics_dict: dict[str, float],
    num_rounds: int,
    out_dir: str,
) -> None:
    tracker[0] += 1
    for k, v in metrics_dict.items():
        results.setdefault(k, []).append(v)
    if tracker[0] >= num_rounds:
        os.makedirs(out_dir, exist_ok=True)
        for name, vals in results.items():
            filepath = os.path.join(out_dir, f"{name}.csv")
            list_to_csv(l=vals, filepath=filepath, column_name=name)


def build_fit_metrics_fn(num_rounds: int, out_dir: str) -> MetricsAggregationFn:
    """Return a fit-metrics aggregation function that writes CSV at end of training."""
    _results: dict[str, list] = {}
    _tracker = [0]

    def fn(metrics):
        agg = {"weighted_avg_loss": 0.0}
        total = 0
        for n, m in metrics:
            agg["weighted_avg_loss"] += m["loss"] * n
            total += n
        if total:
            agg["weighted_avg_loss"] /= total
        _collect_fit_results(_results, _tracker, agg, num_rounds, out_dir)
        return agg

    return fn


def build_eval_metrics_fn(num_rounds: int, out_dir: str) -> MetricsAggregationFn:
    """Return an evaluate-metrics aggregation function that writes CSV at end of training."""
    _results: dict[str, list] = {}
    _tracker = [0]
    keys = ["accuracy", "loss", "precision_score", "recall_score", "f1_score"]

    def fn(metrics):
        agg = {f"weighted_avg_{k}": 0.0 for k in keys}
        total = 0
        for n, m in metrics:
            for k in keys:
                if k in m:
                    agg[f"weighted_avg_{k}"] += m[k] * n
            total += n
        if total:
            for k in keys:
                agg[f"weighted_avg_{k}"] /= total
        
        round_num = _tracker[0] + 1
        acc = agg.get("weighted_avg_accuracy", 0.0)
        print(f"\n[Round {round_num}] Global Evaluation Accuracy: {acc:.4f}\n")
        
        _collect_eval_results(_results, _tracker, agg, num_rounds, out_dir)
        return agg

    return fn


# ---------------------------------------------------------------------------
# Aggregation timing mixin
# ---------------------------------------------------------------------------

class TimingMixin:
    """Mixin that records per-round aggregation latency and writes a CSV."""

    def __init__(self, server_rounds: int, out_dir: str = "results"):
        self._timing_server_rounds = server_rounds
        self._timing_out_dir = out_dir
        self._aggregation_times: dict[int, float] = {}

    def _record_aggregation_time(self, server_round: int, elapsed: float) -> None:
        self._aggregation_times[server_round] = elapsed
        if server_round == self._timing_server_rounds:
            latency_dir = os.path.join(self._timing_out_dir, "latency")
            os.makedirs(latency_dir, exist_ok=True)
            filepath = os.path.join(latency_dir, "aggregation_time.csv")
            dict_to_csv(self._aggregation_times, filepath=filepath,
                        column_name="Aggregation Time (s)")


class BaselineRobustnessMixin:
    """Mixin for baselines to record reputation/trust scores and malicious detection metrics."""

    def __init__(self, server_rounds: int, out_dir: str):
        self._robust_server_rounds = server_rounds
        self._robust_out_dir = out_dir
        # Store round-level metrics
        self._reputation_history: Dict[int, Dict[str, float]] = {}
        self._robustness_metrics: Dict[int, Dict[str, float]] = {}

    def _record_scores(self, server_round: int, scores: Dict[str, float], score_name: str = "reputation"):
        """Record per-client scores for the current round."""
        self._reputation_history[server_round] = scores
        # Also print to console as requested
        print(f"\n[Round {server_round}] {score_name.capitalize()} Scores:")
        for cid, score in sorted(scores.items()):
            print(f"  Client {cid}: {score:.4f}")

    def _compute_and_record_robustness(
        self, 
        server_round: int, 
        results: List[tuple[ClientProxy, FitRes]], 
        classifications: Dict[str, int]
    ):
        """
        Compute detection metrics (TPR, FPR, Accuracy) and record them.
        classifications: dict mapping CID to 1 (Trusted/Benign) or 0 (Untrusted/Malicious)
        """
        tp = fp = tn = fn = 0
        
        for client, fit_res in results:
            cid = client.cid
            is_malicious = "MALICIOUS" in str(fit_res.metrics.get("role", "")).upper()
            pred_trusted = classifications.get(cid, 1) # Default to trusted if not specified
            
            if not is_malicious: # Ground Truth: Benign
                if pred_trusted:
                    tp += 1 # Correctly identified as benign (Trusted)
                else:
                    fn += 1 # Wrongly identified as malicious (Untrusted)
            else: # Ground Truth: Malicious
                if pred_trusted:
                    fp += 1 # Wrongly identified as benign (Trusted)
                else:
                    tn += 1 # Correctly identified as malicious (Untrusted)

        total = len(results)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        accuracy = (tp + tn) / total if total > 0 else 1.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall/tpr": recall,
            "fpr": fpr,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn
        }
        self._robustness_metrics[server_round] = metrics

        if server_round == self._robust_server_rounds:
            self._save_robustness_data()

    def _save_robustness_data(self):
        """Save robustness and reputation data to CSV files."""
        # Save robustness metrics
        robust_dir = os.path.join(self._robust_out_dir, "robustness")
        os.makedirs(robust_dir, exist_ok=True)
        
        # Convert to per-metric dictionaries for dict_to_csv
        for metric_name in ["accuracy", "precision", "recall/tpr", "fpr"]:
            data = {r: m[metric_name] for r, m in self._robustness_metrics.items()}
            filepath = os.path.join(robust_dir, f"{metric_name.replace('/', '_')}.csv")
            dict_to_csv(data, filepath, column_name=metric_name)

        # Save reputation scores history
        rep_dir = os.path.join(self._robust_out_dir, "reputation")
        os.makedirs(rep_dir, exist_ok=True)
        
        # We'll save a file per round for reputation if there are many, 
        # but let's just do a consolidated one or similar for now.
        # Actually, let's just save the round-level scores in a way the user can easily see.
        for r, scores in self._reputation_history.items():
            filepath = os.path.join(rep_dir, f"round_{r}_scores.csv")
            dict_to_csv(scores, filepath, column_name="score")


# ---------------------------------------------------------------------------
# Gradient flattening helpers (re-exported for convenience)
# ---------------------------------------------------------------------------

def flatten_ndarrays(ndarrays: list[np.ndarray]) -> np.ndarray:
    """Concatenate all ndarrays into a single 1-D vector."""
    return np.concatenate([arr.ravel() for arr in ndarrays])


def unflatten_ndarrays(flat: np.ndarray, template: list[np.ndarray]) -> list[np.ndarray]:
    """Reconstruct a list of ndarrays matching *template* shapes from a flat vector."""
    result = []
    idx = 0
    for arr in template:
        numel = arr.size
        result.append(flat[idx: idx + numel].reshape(arr.shape))
        idx += numel
    return result
