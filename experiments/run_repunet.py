#!/usr/bin/env python3
"""
experiments/run_repunet.py
---------------------------
RepuNet reputation-based aggregation experiment.

Usage:
    python -m experiments.run_repunet [options]
    python -m experiments.run_repunet --help
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from experiments._base import build_arg_parser, run_experiment
from baselines.repunet import RepuNetStrategy


def make_strategy(args, initial_params, num_rounds, fit_fn, eval_fn):
    return RepuNetStrategy(
        server_rounds=num_rounds,
        ema_decay=getattr(args, "ema_decay", 0.9),
        init_reputation=getattr(args, "init_reputation", 0.5),
        min_reputation=getattr(args, "min_reputation", 0.05),
        out_dir=args.out_dir or "results/repunet",
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
    parser = build_arg_parser("RepuNet baseline experiment")
    parser.add_argument(
        "--ema-decay", type=float, default=0.9,
        help="EMA decay factor for reputation scores (default: 0.9)",
    )
    parser.add_argument(
        "--init-reputation", type=float, default=0.5,
        help="Initial reputation for new clients (default: 0.5)",
    )
    parser.add_argument(
        "--min-reputation", type=float, default=0.05,
        help="Floor for reputation scores (default: 0.05)",
    )
    args = parser.parse_args()
    run_experiment(args, make_strategy, strategy_name="repunet")
