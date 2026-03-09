"""
baselines/repunet.py
---------------------
RepuNet: Reputation-Network-Based Robust Federated Aggregation.

Concept
-------
RepuNet maintains a per-client *reputation score* that is updated each round
based on how similar each client's update is to the (current round's)
consensus gradient.  Updates are included in aggregation proportionally to
their reputation.

Update rule
-----------
  cos_i  = cosine_similarity(g_i, g_mean_benign)
  rep_i  ← ema_decay * rep_i + (1 − ema_decay) * ReLU(cos_i)
  (new clients start with rep = 0.5)

Aggregation
-----------
  w_i  = rep_i * num_examples_i  (normalised)
  g*   = Σ w_i * g_i

This lightweight approach avoids PCA, Mahalanobis distance or clustering and
is very fast even for large numbers of clients.
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


class RepuNetStrategy(TimingMixin, BaselineRobustnessMixin, FedAvg):
    """Reputation-network-based robust aggregation.

    Additional constructor parameters
    ----------------------------------
    server_rounds : int
        Total rounds for timing CSV flush.
    ema_decay : float
        Exponential moving-average decay factor for reputation scores.
        Higher = more stable / slower adaptation.  Default: ``0.9``.
    init_reputation : float
        Initial reputation for newly seen clients.  Default: ``0.5``.
    min_reputation : float
        Floor for reputation (prevents complete exclusion on first bad round).
        Default: ``0.05``.
    out_dir : str
        Output directory.  Default: ``"results/repunet"``.
    """

    def __init__(
        self,
        server_rounds: int = 10,
        ema_decay: float = 0.9,
        init_reputation: float = 0.5,
        min_reputation: float = 0.05,
        out_dir: str = "results/repunet",
        **kwargs,
    ):
        FedAvg.__init__(self, **kwargs)
        TimingMixin.__init__(self, server_rounds=server_rounds, out_dir=out_dir)
        BaselineRobustnessMixin.__init__(self, server_rounds=server_rounds, out_dir=out_dir)
        self.ema_decay = ema_decay
        self.init_reputation = init_reputation
        self.min_reputation = min_reputation
        # cid → reputation score
        self._reputations: dict[str, float] = {}

    # ------------------------------------------------------------------
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

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

        cids: list[str] = []
        flat_updates: list[np.ndarray] = []
        sample_counts: list[int] = []

        for client, fit_res in results:
            cid = client.cid
            cids.append(cid)
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            flat_updates.append(flatten_ndarrays(ndarrays))
            sample_counts.append(fit_res.num_examples)

            # Initialise reputation for new clients
            if cid not in self._reputations:
                self._reputations[cid] = self.init_reputation

        # 1. Adaptive Norm Clipping
        matrix = np.stack(flat_updates, axis=0)  # (n, d)
        norms = np.linalg.norm(matrix, axis=1)
        c = np.median(norms)
        if c < 1e-12:
            c = 1.0
        
        scales = np.minimum(1.0, c / (norms + 1e-12))
        matrix = matrix * scales[:, np.newaxis]

        # 2. Robust Consensus: Coordinate-wise Median
        # Using median is more Byzantine-robust than a weighted mean for the reference gradient
        g_mean = np.median(matrix, axis=0)

        # 3. Update reputations via cosine similarity to robust consensus
        for i, cid in enumerate(cids):
            cos = self._cosine_similarity(matrix[i], g_mean)
            new_signal = max(0.0, cos)  # ReLU
            self._reputations[cid] = (
                self.ema_decay * self._reputations[cid]
                + (1.0 - self.ema_decay) * new_signal
            )
            self._reputations[cid] = max(self._reputations[cid], self.min_reputation)

        # Re-compute weights with updated reputations
        updated_reps = np.array([self._reputations[cid] for cid in cids])
        
        # 4. Recording scores and classifications
        self._record_scores(server_round, {cid: float(self._reputations[cid]) for cid in cids}, score_name="reputation")
        # Classification: Trusted if reputation is significantly above minimum
        classifications = {cid: (1 if self._reputations[cid] > self.min_reputation * 1.5 else 0) for cid in cids}
        self._compute_and_record_robustness(server_round, results, classifications)

        counts = np.array(sample_counts, dtype=float)
        weights = updated_reps * counts
        w_sum = weights.sum()
        if w_sum < 1e-12:
            weights = counts / counts.sum()
        else:
            weights /= w_sum

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
