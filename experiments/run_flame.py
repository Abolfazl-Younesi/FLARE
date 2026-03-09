#!/usr/bin/env python3
"""
experiments/run_flame.py
-------------------------
FLAME baseline experiment.

Usage:
    python -m experiments.run_flame [options]
    python -m experiments.run_flame --help
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from experiments._base import build_arg_parser, run_experiment
from baselines.flame import FLAMEStrategy


def make_strategy(args, initial_params, num_rounds, fit_fn, eval_fn):
    return FLAMEStrategy(
        server_rounds=num_rounds,
        noise_multiplier=getattr(args, "noise_multiplier", 0.001),
        min_cluster_size=getattr(args, "min_cluster_size", 2),
        out_dir=args.out_dir or "results/flame",
        initial_parameters=initial_params,
        fit_metrics_aggregation_fn=fit_fn,
        evaluate_metrics_aggregation_fn=eval_fn,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_available_clients=1,
        min_fit_clients=1,
        min_evaluate_clients=1,
        on_fit_config_fn=lambda r: {"send_time": time.time(), "dataset": args.dataset},
        on_evaluate_config_fn=lambda r: {"dataset": args.dataset},
    )


if __name__ == "__main__":
    parser = build_arg_parser("FLAME baseline experiment")
    parser.add_argument(
        "--noise-multiplier", type=float, default=0.001,
        help="Gaussian noise scale relative to clipping bound (default: 0.001)",
    )
    parser.add_argument(
        "--min-cluster-size", type=int, default=2,
        help="HDBSCAN minimum cluster size (default: 2)",
    )
    args = parser.parse_args()
    run_experiment(args, make_strategy, strategy_name="flame")
