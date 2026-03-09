#!/usr/bin/env python3
"""
experiments/run_trimmed_mean.py
--------------------------------
Coordinate-wise Trimmed Mean baseline experiment.

Usage:
    python -m experiments.run_trimmed_mean [options]
    python -m experiments.run_trimmed_mean --help
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from experiments._base import build_arg_parser, run_experiment
from baselines.aggr_strategies import TrimmedMeanStrategy


def make_strategy(args, initial_params, num_rounds, fit_fn, eval_fn):
    return TrimmedMeanStrategy(
        server_rounds=num_rounds,
        trim_fraction=getattr(args, "trim_fraction", 0.1),
        out_dir=args.out_dir or "results/trimmed_mean",
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
    parser = build_arg_parser("Trimmed Mean baseline experiment")
    parser.add_argument(
        "--trim-fraction", type=float, default=0.1,
        help="Fraction of extremes to trim per coordinate (default: 0.1)",
    )
    args = parser.parse_args()
    run_experiment(args, make_strategy, strategy_name="trimmed_mean")
