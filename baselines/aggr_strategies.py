"""
baselines/aggr_strategies.py
-----------------------------
Flower-compatible aggregation strategies:

  - FedAvgBaseline   : Standard FedAvg with ALIE hook + timing CSV
  - KrumStrategy     : Byzantine-robust single-best-client selection (Krum)
  - MultiKrumStrategy: Byzantine-robust top-K selection (Multi-Krum)
  - TrimmedMeanStrategy: Coordinate-wise trimmed mean

All strategies accept the same constructor arguments as flwr.server.strategy.FedAvg
plus a few extras documented below.
"""
from __future__ import annotations

import os
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


# ---------------------------------------------------------------------------
# FedAvg Baseline
# ---------------------------------------------------------------------------

class FedAvgBaseline(TimingMixin, BaselineRobustnessMixin, FedAvg):
    """Standard FedAvg with ALIE malicious-network hook and aggregation-timing CSV.

    Additional constructor parameters
    ----------------------------------
    server_rounds : int
        Total number of FL rounds (used to know when to flush timing CSV).
    out_dir : str
        Root directory for result CSVs.  Default: ``"results/fedavg"``.
    """

    def __init__(self, server_rounds: int = 10, out_dir: str = "results/fedavg", **kwargs):
        FedAvg.__init__(self, **kwargs)
        TimingMixin.__init__(self, server_rounds=server_rounds, out_dir=out_dir)
        BaselineRobustnessMixin.__init__(self, server_rounds=server_rounds, out_dir=out_dir)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        # Apply ALIE hook before any aggregation
        alie_malicious_network_hook(results)
        t0 = time.time()

        # Extract and flatten updates
        flat_updates = []
        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            flat_updates.append(flatten_ndarrays(ndarrays))

        # Adaptive Norm Clipping: Median-based
        norms = [np.linalg.norm(upd) for upd in flat_updates]
        c = np.median(norms)
        if c < 1e-12:
            c = 1.0

        clipped_results = []
        for i, (client, fit_res) in enumerate(results):
            if norms[i] > c:
                ndarrays = parameters_to_ndarrays(fit_res.parameters)
                scale = c / norms[i]
                scaled_ndarrays = [arr * scale for arr in ndarrays]
                fit_res.parameters = ndarrays_to_parameters(scaled_ndarrays)
            clipped_results.append((client, fit_res))

        aggregated = FedAvg.aggregate_fit(self, server_round, clipped_results, failures)
        
        # Classification: FedAvg trusts everyone
        classifications = {client.cid: 1 for client, _ in results}
        self._compute_and_record_robustness(server_round, results, classifications)

        self._record_aggregation_time(server_round, time.time() - t0)
        return aggregated


# ---------------------------------------------------------------------------
# Krum
# ---------------------------------------------------------------------------

class KrumStrategy(TimingMixin, BaselineRobustnessMixin, FedAvg):
    """Byzantine-robust aggregation via Krum.

    Krum selects the single update that is closest (in L2 norm) to the sum
    of its (n - f - 2) nearest neighbours, where *f* is the assumed number of
    Byzantine clients.  The global model is set to that single client's update.

    Additional constructor parameters
    ----------------------------------
    server_rounds : int
        Total number of FL rounds.
    num_byzantine : int
        Number of Byzantine clients assumed to be present.
    out_dir : str
        Root directory for result CSVs.  Default: ``"results/krum"``.
    """

    def __init__(
        self,
        server_rounds: int = 10,
        num_byzantine: int = 0,
        out_dir: str = "results/krum",
        **kwargs,
    ):
        FedAvg.__init__(self, **kwargs)
        TimingMixin.__init__(self, server_rounds=server_rounds, out_dir=out_dir)
        BaselineRobustnessMixin.__init__(self, server_rounds=server_rounds, out_dir=out_dir)
        self.num_byzantine = num_byzantine

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

        # Flatten all updates
        updates: list[np.ndarray] = []
        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            updates.append(flatten_ndarrays(ndarrays))

        n = len(updates)
        f = self.num_byzantine
        k = n - f - 2  # number of neighbours to consider
        k = max(1, min(k, n - 1))

        # Compute pairwise squared distances
        scores = np.zeros(n)
        for i in range(n):
            dists = sorted(
                np.sum((updates[i] - updates[j]) ** 2) for j in range(n) if j != i
            )
            scores[i] = sum(dists[:k])

        best_idx = int(np.argmin(scores))
        
        # Recording scores and classifications
        # Krum score: lower is better/more trusted. We'll report as 1/score for visualization consistency if needed, 
        # but let's just report raw and classify
        self._record_scores(server_round, {results[i][0].cid: float(scores[i]) for i in range(n)}, score_name="krum")
        classifications = {results[i][0].cid: (1 if i == best_idx else 0) for i in range(n)}
        self._compute_and_record_robustness(server_round, results, classifications)

        best_ndarrays = parameters_to_ndarrays(results[best_idx][1].parameters)
        aggregated_params = ndarrays_to_parameters(best_ndarrays)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        self._record_aggregation_time(server_round, time.time() - t0)
        return aggregated_params, metrics_aggregated


# ---------------------------------------------------------------------------
# Multi-Krum
# ---------------------------------------------------------------------------

class MultiKrumStrategy(TimingMixin, BaselineRobustnessMixin, FedAvg):
    """Byzantine-robust aggregation via Multi-Krum.

    Multi-Krum selects the top-K updates by Krum score and averages them.

    Additional constructor parameters
    ----------------------------------
    server_rounds : int
        Total number of FL rounds.
    num_byzantine : int
        Number of Byzantine clients assumed.  Default: 0.
    top_k : int or None
        Number of best updates to average.  If None, defaults to n - f.
    out_dir : str
        Root directory for result CSVs.  Default: ``"results/multikrum"``.
    """

    def __init__(
        self,
        server_rounds: int = 10,
        num_byzantine: int = 0,
        top_k: Optional[int] = None,
        out_dir: str = "results/multikrum",
        **kwargs,
    ):
        FedAvg.__init__(self, **kwargs)
        TimingMixin.__init__(self, server_rounds=server_rounds, out_dir=out_dir)
        BaselineRobustnessMixin.__init__(self, server_rounds=server_rounds, out_dir=out_dir)
        self.num_byzantine = num_byzantine
        self.top_k = top_k

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

        updates: list[np.ndarray] = []
        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            updates.append(flatten_ndarrays(ndarrays))

        n = len(updates)
        f = self.num_byzantine
        k_neighbours = max(1, min(n - f - 2, n - 1))
        top_k = self.top_k if self.top_k is not None else max(1, n - f)
        top_k = min(top_k, n)

        # Krum scores
        scores = np.zeros(n)
        for i in range(n):
            dists = sorted(
                np.sum((updates[i] - updates[j]) ** 2) for j in range(n) if j != i
            )
            scores[i] = sum(dists[:k_neighbours])

        selected_indices = np.argsort(scores)[:top_k]

        # Recording scores and classifications
        self._record_scores(server_round, {results[i][0].cid: float(scores[i]) for i in range(n)}, score_name="multikrum")
        classifications = {results[i][0].cid: (1 if i in selected_indices else 0) for i in range(n)}
        self._compute_and_record_robustness(server_round, results, classifications)

        # Weighted average of selected updates
        total_examples = sum(results[i][1].num_examples for i in selected_indices)
        template_ndarrays = parameters_to_ndarrays(results[0][1].parameters)
        agg_flat = np.zeros_like(flatten_ndarrays(template_ndarrays))

        for idx in selected_indices:
            w = results[idx][1].num_examples / total_examples
            ndarrays = parameters_to_ndarrays(results[idx][1].parameters)
            agg_flat += w * flatten_ndarrays(ndarrays)

        new_ndarrays = unflatten_ndarrays(agg_flat, template_ndarrays)
        aggregated_params = ndarrays_to_parameters(new_ndarrays)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        self._record_aggregation_time(server_round, time.time() - t0)
        return aggregated_params, metrics_aggregated


# ---------------------------------------------------------------------------
# Trimmed Mean
# ---------------------------------------------------------------------------

class TrimmedMeanStrategy(TimingMixin, BaselineRobustnessMixin, FedAvg):
    """Coordinate-wise trimmed mean aggregation.

    For each model parameter coordinate, the *trim_fraction* fraction of
    highest and lowest values are discarded before computing the mean.

    Additional constructor parameters
    ----------------------------------
    server_rounds : int
        Total number of FL rounds.
    trim_fraction : float
        Fraction of updates to trim from each tail (0 ≤ trim_fraction < 0.5).
        E.g., 0.1 trims the bottom 10 % and top 10 % of values per coordinate.
    out_dir : str
        Root directory for result CSVs.  Default: ``"results/trimmed_mean"``.
    """

    def __init__(
        self,
        server_rounds: int = 10,
        trim_fraction: float = 0.1,
        out_dir: str = "results/trimmed_mean",
        **kwargs,
    ):
        FedAvg.__init__(self, **kwargs)
        TimingMixin.__init__(self, server_rounds=server_rounds, out_dir=out_dir)
        BaselineRobustnessMixin.__init__(self, server_rounds=server_rounds, out_dir=out_dir)
        self.trim_fraction = trim_fraction

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

        # Stack all flat update vectors  shape: (n_clients, n_params)
        flat_updates: list[np.ndarray] = []
        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            flat_updates.append(flatten_ndarrays(ndarrays))

        # Adaptive Norm Clipping
        matrix = np.stack(flat_updates, axis=0)  # (n, d)
        norms = np.linalg.norm(matrix, axis=1)
        c = np.median(norms)
        if c < 1e-12:
            c = 1.0
        
        # Apply clipping to matrix
        scales = np.minimum(1.0, c / (norms + 1e-12))
        matrix = matrix * scales[:, np.newaxis]

        n = matrix.shape[0]
        k = max(1, int(np.floor(self.trim_fraction * n)))

        if 2 * k >= n:
            # Fallback: just average everything
            agg_flat = matrix.mean(axis=0)
        else:
            sorted_matrix = np.sort(matrix, axis=0)
            trimmed = sorted_matrix[k: n - k, :]
            agg_flat = trimmed.mean(axis=0)

        # Classification for Trimmed Mean: 
        # For simplicity, we can't easily map back per-client per-coordinate, 
        # but we can see which clients were excluded in MOST coordinates or similar.
        # Alternatively, we just report ground truth vs the fact that it's a "robust" average.
        # Trimmed mean usually assumes some % are malicious.
        # Let's just mark everyone as "trusted" since they all contribute to the mean (after trimming).
        classifications = {client.cid: 1 for client, _ in results}
        self._compute_and_record_robustness(server_round, results, classifications)

        template_ndarrays = parameters_to_ndarrays(results[0][1].parameters)
        new_ndarrays = unflatten_ndarrays(agg_flat, template_ndarrays)
        aggregated_params = ndarrays_to_parameters(new_ndarrays)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        self._record_aggregation_time(server_round, time.time() - t0)
        return aggregated_params, metrics_aggregated
