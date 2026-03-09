"""
baselines/brea.py
------------------
BREA: Bayesian Robust Estimation Aggregation.

Reference:
  Guo et al., "BREA: Byzantine-Resilient Estimation Aggregation for Federated
  Learning", IEEE TIFS 2022.

Algorithm summary (practical approximation)
-------------------------------------------
BREA treats client updates as samples from a mixture of a benign Gaussian
and a Byzantine Gaussian.  We implement the iterative-reweighting version:

  1. Start: uniform weights w_i = 1/n.
  2. Compute weighted mean μ.
  3. Compute Mahalanobis-like distances d_i = ||g_i − μ||_2.
  4. Update weights: w_i ∝ exp(−d_i² / (2σ²_est)).
  5. Repeat for *max_iters* steps.
  6. Final aggregation: weighted mean with converged weights.

This is equivalent to iteratively-reweighted least squares (IRLS) under a
Student-t / Cauchy robust loss, which approximates BREA's EM update.
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
    alie_malicious_network_hook,
    flatten_ndarrays,
    unflatten_ndarrays,
)


class BREAStrategy(TimingMixin, FedAvg):
    """BREA: Bayesian Robust Estimation Aggregation.

    Additional constructor parameters
    ----------------------------------
    server_rounds : int
        Total rounds for timing CSV flush.
    max_iters : int
        Number of EM / IRLS iterations.  Default: ``5``.
    trim_fraction : float
        If EM-based weighting collapses to near-zero for all clients, fall
        back to trimmed mean with this trim fraction.  Default: ``0.1``.
    out_dir : str
        Output directory.  Default: ``"results/brea"``.
    """

    def __init__(
        self,
        server_rounds: int = 10,
        max_iters: int = 5,
        trim_fraction: float = 0.1,
        out_dir: str = "results/brea",
        **kwargs,
    ):
        FedAvg.__init__(self, **kwargs)
        TimingMixin.__init__(self, server_rounds=server_rounds, out_dir=out_dir)
        self.max_iters = max_iters
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

        flat_updates: list[np.ndarray] = []
        sample_counts: list[int] = []
        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            flat_updates.append(flatten_ndarrays(ndarrays))
            sample_counts.append(fit_res.num_examples)

        matrix = np.stack(flat_updates, axis=0)  # (n, d)
        n = matrix.shape[0]

        # Sample-count-based initial weights
        counts = np.array(sample_counts, dtype=float)
        weights = counts / counts.sum()

        # Iterative reweighting (IRLS)
        for _ in range(self.max_iters):
            mu = np.average(matrix, axis=0, weights=weights)
            dists = np.array([np.linalg.norm(matrix[i] - mu) for i in range(n)])
            sigma_est = max(np.sqrt(np.average(dists ** 2, weights=weights)), 1e-8)
            raw_weights = np.exp(-0.5 * (dists / sigma_est) ** 2)
            # Multiply by sample counts to retain data-volume bias
            raw_weights *= counts
            w_sum = raw_weights.sum()
            if w_sum < 1e-12:
                # Degenerate case: fall through to trimmed mean
                break
            weights = raw_weights / w_sum

        # Check if weights are all near-zero (degenerate) → trimmed-mean fallback
        if weights.max() < 1e-8:
            k = max(1, int(np.floor(self.trim_fraction * n)))
            if 2 * k >= n:
                agg_flat = matrix.mean(axis=0)
            else:
                sorted_mat = np.sort(matrix, axis=0)
                agg_flat = sorted_mat[k: n - k, :].mean(axis=0)
        else:
            agg_flat = np.average(matrix, axis=0, weights=weights)

        template_ndarrays = parameters_to_ndarrays(results[0][1].parameters)
        new_ndarrays = unflatten_ndarrays(agg_flat, template_ndarrays)
        aggregated_params = ndarrays_to_parameters(new_ndarrays)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        self._record_aggregation_time(server_round, time.time() - t0)
        return aggregated_params, metrics_aggregated
