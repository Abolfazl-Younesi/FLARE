"""
baselines/fltrust.py
---------------------
FLTrust: FL Trust Score-based Robust Aggregation.

Reference:
  Cao et al., "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping",
  NDSS 2022.  https://arxiv.org/abs/2012.13995

Algorithm summary
-----------------
The server holds a small, clean "root" dataset. Each round:
1. The server computes a gradient ``g_0`` on the root dataset using the
   current global model.
2. Each client gradient ``g_i`` receives a *trust score* proportional to
   the ReLU of the cosine similarity between ``g_i`` and ``g_0``.
3. Each ``g_i`` is normalized to have the same magnitude as ``g_0``.
4. The new global model is the trust-weighted average of normalized updates
   plus the current global model.

In the Flower simulation context (where clients send *weights*, not raw
gradients) we treat the sent parameters as weight-level updates and compute
the pseudo-gradient as ``params_client - params_global``.
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


class FLTrustStrategy(TimingMixin, BaselineRobustnessMixin, FedAvg):
    """FLTrust aggregation strategy.

    The "server root dataset" is approximated by averaging all client updates
    once per round before computing trust scores (a practical approximation
    used when the server has no labelled data).  Alternatively, users can
    subclass and override ``_compute_root_gradient`` to inject a real server-
    side gradient.

    Additional constructor parameters
    ----------------------------------
    server_rounds : int
        Total rounds for timing CSV flush.
    out_dir : str
        Output directory.  Default: ``"results/fltrust"``.
    """

    def __init__(
        self,
        server_rounds: int = 10,
        out_dir: str = "results/fltrust",
        **kwargs,
    ):
        FedAvg.__init__(self, **kwargs)
        TimingMixin.__init__(self, server_rounds=server_rounds, out_dir=out_dir)
        BaselineRobustnessMixin.__init__(self, server_rounds=server_rounds, out_dir=out_dir)
        # Store current global parameters between rounds
        self._global_ndarrays: Optional[list[np.ndarray]] = None

    # ------------------------------------------------------------------
    # Override configure_fit to cache the current global parameters
    # ------------------------------------------------------------------

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _compute_root_gradient(
        self, flat_updates: list[np.ndarray]
    ) -> np.ndarray:
        """Return a reference gradient used to produce trust scores.

        Default implementation: coordinate-wise median across all clients
        (a neutral, Byzantine-robust reference).  Subclass and override to
        inject a real server-side gradient.
        """
        matrix = np.stack(flat_updates, axis=0)  # (n, d)
        return np.median(matrix, axis=0)

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

        # Flatten all client updates
        flat_updates: list[np.ndarray] = []
        num_examples: list[int] = []
        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            flat_updates.append(flatten_ndarrays(ndarrays))
            num_examples.append(fit_res.num_examples)

        # 1. Adaptive Norm Clipping
        matrix = np.stack(flat_updates, axis=0)  # (n, d)
        norms = np.linalg.norm(matrix, axis=1)
        c = np.median(norms)
        if c < 1e-12:
            c = 1.0
        
        scales = np.minimum(1.0, c / (norms + 1e-12))
        matrix = matrix * scales[:, np.newaxis]
        # Update flat_updates with clipped versions
        flat_updates = [matrix[i] for i in range(len(flat_updates))]

        # 2. Compute reference gradient
        g0 = self._compute_root_gradient(flat_updates)
        g0_norm = np.linalg.norm(g0)

        # 3. Trust scores: ReLU(cosine_sim(g_i, g0))
        trust_scores = np.array(
            [max(0.0, self._cosine_similarity(g, g0)) for g in flat_updates]
        )

        # 4. Recording scores and classifications
        self._record_scores(server_round, {results[i][0].cid: float(trust_scores[i]) for i in range(len(results))}, score_name="trust")
        # Classification: Trusted if trust score > 0
        classifications = {results[i][0].cid: (1 if trust_scores[i] > 0 else 0) for i in range(len(results))}
        self._compute_and_record_robustness(server_round, results, classifications)

        # Normalize each client update to magnitude of g0, then weight by trust
        trust_sum = trust_scores.sum()
        if trust_sum < 1e-12:
            # All trust scores zero → fallback to plain average
            trust_scores = np.ones(len(flat_updates)) / len(flat_updates)
            trust_sum = 1.0

        agg_flat = np.zeros_like(flat_updates[0])
        for i, g_i in enumerate(flat_updates):
            g_i_norm = np.linalg.norm(g_i)
            if g_i_norm > 1e-12:
                normalized = g_i * (g0_norm / g_i_norm)
            else:
                normalized = g_i
            agg_flat += (trust_scores[i] / trust_sum) * normalized

        template_ndarrays = parameters_to_ndarrays(results[0][1].parameters)
        new_ndarrays = unflatten_ndarrays(agg_flat, template_ndarrays)
        aggregated_params = ndarrays_to_parameters(new_ndarrays)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        self._record_aggregation_time(server_round, time.time() - t0)
        return aggregated_params, metrics_aggregated
