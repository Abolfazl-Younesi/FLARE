"""
baselines/flame.py
-------------------
FLAME: Taming Backdoors in Federated Learning.

Reference:
  Nguyen et al., "FLAME: Taming Backdoors in Federated Learning",
  USENIX Security 2022.  https://arxiv.org/abs/2101.02281

Algorithm summary
-----------------
1. Compute cosine similarities among all client updates.
2. Cluster updates using HDBSCAN on the cosine-distance matrix.
3. Retain only the largest cluster (assumed to be benign).
4. Clip each retained update to a clipping bound S = median(||g_i||).
5. Add calibrated Gaussian noise σ = λ·S (noise multiplier λ).
6. Average retained+noised updates.

Dependencies
------------
  pip install hdbscan   (or scikit-learn>=1.3 which ships with HDBSCAN)

We fall back to scikit-learn's HDBSCAN (sklearn.cluster.HDBSCAN) and, if
that is also unavailable, to a cosine-distance-based k-means fallback.
"""
from __future__ import annotations

import time
from typing import Optional, Union

import numpy as np

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from baselines.common import (
    TimingMixin,
    BaselineRobustnessMixin,
    alie_malicious_network_hook,
    flatten_ndarrays,
    unflatten_ndarrays,
)


def _try_hdbscan_cluster(matrix: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Cluster rows of *matrix* using HDBSCAN; returns label array.

    Falls back to a simple 2-class split by cosine distance if HDBSCAN is
    not available.
    """
    # Cosine distance matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1e-12, norms)
    normed = matrix / norms
    cos_sim = normed @ normed.T
    dist_matrix = np.clip(1.0 - cos_sim, 0.0, 2.0)

    try:
        from sklearn.cluster import HDBSCAN as SklHDBSCAN
        clusterer = SklHDBSCAN(min_cluster_size=min_cluster_size, metric="precomputed")
        labels = clusterer.fit_predict(dist_matrix.astype(np.float64))
        return labels
    except Exception:
        pass

    try:
        import hdbscan as hdbscan_lib
        clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=min_cluster_size, metric="precomputed")
        labels = clusterer.fit_predict(dist_matrix.astype(np.float64))
        return labels
    except Exception:
        pass

    # Fallback: split at median cosine distance from the global mean
    mean_vec = normed.mean(axis=0)
    mean_vec /= max(np.linalg.norm(mean_vec), 1e-12)
    sims = normed @ mean_vec
    threshold = np.median(sims)
    labels = np.where(sims >= threshold, 0, -1)
    return labels


class FLAMEStrategy(TimingMixin, BaselineRobustnessMixin, FedAvg):
    """FLAME: Byzantine-robust aggregation via HDBSCAN clustering + noise injection.

    Additional constructor parameters
    ----------------------------------
    server_rounds : int
        Total rounds for timing CSV flush.
    noise_multiplier : float
        Gaussian noise scale relative to the clipping bound S.
        FLAME paper uses λ ≈ 0.001.  Default: ``0.001``.
    min_cluster_size : int
        Minimum cluster size for HDBSCAN.  Default: ``2``.
    out_dir : str
        Output directory.  Default: ``"results/flame"``.
    """

    def __init__(
        self,
        server_rounds: int = 10,
        noise_multiplier: float = 0.001,
        min_cluster_size: int = 2,
        out_dir: str = "results/flame",
        **kwargs,
    ):
        FedAvg.__init__(self, **kwargs)
        TimingMixin.__init__(self, server_rounds=server_rounds, out_dir=out_dir)
        BaselineRobustnessMixin.__init__(self, server_rounds=server_rounds, out_dir=out_dir)
        self.noise_multiplier = noise_multiplier
        self.min_cluster_size = min_cluster_size

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        alie_malicious_network_hook(results)
        t0 = time.time()

        flat_updates: list[np.ndarray] = []
        num_examples_list: list[int] = []
        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            flat_updates.append(flatten_ndarrays(ndarrays))
            num_examples_list.append(fit_res.num_examples)

        matrix = np.stack(flat_updates, axis=0)  # (n, d)
        n = matrix.shape[0]

        # --- Clustering ---
        min_cs = max(2, min(self.min_cluster_size, n // 2))
        labels = _try_hdbscan_cluster(matrix, min_cluster_size=min_cs)

        # Find largest non-noise cluster
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        if len(unique_labels) == 0:
            # All marked noise → use all
            selected_mask = np.ones(n, dtype=bool)
        else:
            largest_label = unique_labels[np.argmax(counts)]
            selected_mask = labels == largest_label

        # Classification: Trusted if in the largest cluster
        classifications = {results[i][0].cid: (1 if selected_mask[i] else 0) for i in range(n)}
        self._compute_and_record_robustness(server_round, results, classifications)

        selected_updates = matrix[selected_mask]
        selected_examples = np.array(num_examples_list)[selected_mask]

        # --- Clipping ---
        norms = np.linalg.norm(selected_updates, axis=1)
        S = float(np.median(norms))
        if S < 1e-12:
            S = 1.0

        clipped = []
        for upd in selected_updates:
            norm = np.linalg.norm(upd)
            scale = min(1.0, S / norm) if norm > 1e-12 else 1.0
            clipped.append(upd * scale)

        # --- Weighted average ---
        total = selected_examples.sum()
        if total == 0:
            total = len(clipped)
            weights = np.ones(len(clipped)) / total
        else:
            weights = selected_examples / total

        agg_flat = np.zeros_like(flat_updates[0])
        for w, upd in zip(weights, clipped):
            agg_flat += w * upd

        # --- Gaussian noise injection ---
        sigma = self.noise_multiplier * S
        agg_flat += np.random.normal(0.0, sigma, size=agg_flat.shape)

        template_ndarrays = parameters_to_ndarrays(results[0][1].parameters)
        new_ndarrays = unflatten_ndarrays(agg_flat, template_ndarrays)
        aggregated_params = ndarrays_to_parameters(new_ndarrays)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        self._record_aggregation_time(server_round, time.time() - t0)
        return aggregated_params, metrics_aggregated
