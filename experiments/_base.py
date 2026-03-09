"""
experiments/_base.py
---------------------
Shared argument parser and simulation runner used by all experiment scripts.

Usage pattern in each run_*.py:
    from experiments._base import build_arg_parser, run_experiment
    from baselines.xxx import XxxStrategy

    def make_strategy(args, initial_params, num_rounds, fit_fn, eval_fn):
        return XxxStrategy(
            server_rounds=num_rounds,
            initial_parameters=initial_params,
            fit_metrics_aggregation_fn=fit_fn,
            evaluate_metrics_aggregation_fn=eval_fn,
            fraction_fit=args.fraction_fit,
            fraction_evaluate=args.fraction_evaluate,
            min_available_clients=1, min_fit_clients=1, min_evaluate_clients=1,
            on_fit_config_fn=lambda r: {"send_time": __import__("time").time(), "dataset": args.dataset},
            on_evaluate_config_fn=lambda r: {"dataset": args.dataset},
        )

    if __name__ == "__main__":
        parser = build_arg_parser("FedAvg baseline experiment")
        args = parser.parse_args()
        run_experiment(args, make_strategy, strategy_name="fedavg")
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Ensure project root is on the path when running as a module
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from flwr.common import ndarrays_to_parameters, EventType
from flwr.common.typing import Run
from flwr.common.constant import RUN_ID_NUM_BYTES
from flwr.server.superlink.linkstate.utils import generate_rand_int_from_bytes
from flwr.simulation.run_simulation import _run_simulation as run_simulation
from flwr.server import ServerApp, ServerConfig, ServerAppComponents

from byebye_badclients.client_app import app as client_app
from byebye_badclients.task import get_weights
from byebye_badclients.util import load_model
from baselines.common import (
    iid_level_to_params,
    build_fit_metrics_fn,
    build_eval_metrics_fn,
)


def build_arg_parser(description: str = "FLARE baseline experiment") -> argparse.ArgumentParser:
    """Return an ArgumentParser pre-loaded with common FL experiment arguments."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--rounds", type=int, default=5,
        help="Number of federated learning rounds (default: 5)",
    )
    parser.add_argument(
        "--dataset", type=str, default="mnist",
        choices=["mnist", "cifar10", "svhn", "cifar100"],
        help="Dataset to use (default: mnist)",
    )
    parser.add_argument(
        "--num-clients", type=int, default=10,
        help="Number of virtual SuperNodes / clients (default: 10)",
    )
    parser.add_argument(
        "--local-epochs", type=int, default=5,
        help="Local training epochs per round (default: 5)",
    )
    parser.add_argument(
        "--fraction-fit", type=float, default=0.4,
        help="Fraction of clients selected for training each round (default: 0.4)",
    )
    parser.add_argument(
        "--fraction-evaluate", type=float, default=0.5,
        help="Fraction of clients selected for evaluation (default: 0.5)",
    )
    parser.add_argument(
        "--iid-level", type=float, default=1.0,
        choices=[1.0, 0.7, 0.5, 0.3, 0.1],
        help=(
            "Data heterogeneity level: "
            "1.0=IID, 0.7=slightly non-IID, 0.5=moderate, 0.3=strong, 0.1=extreme "
            "(default: 1.0)"
        ),
    )
    parser.add_argument(
        "--malicious-prob", type=float, default=0.0,
        help="Probability [0,1] that a client is malicious (default: 0.0)",
    )
    parser.add_argument(
        "--attack-patterns", type=str, default="",
        help=(
            "Comma-separated attack patterns to enable.  "
            "Options: label-flipping, random-update, update-scaling, alie, "
            "statistical-mimicry, adaptive-attack.  "
            "Empty string = no attacks (default: empty)."
        ),
    )
    parser.add_argument(
        "--update-scaling-factor", type=float, default=2.0,
        help="Scaling factor used by the update-scaling attack (default: 2.0)",
    )
    parser.add_argument(
        "--total-max-samples", type=int, default=500,
        help="Max training samples per client partition (-1 = all, default: 500)",
    )
    parser.add_argument(
        "--no-defense-fedavg", action="store_true",
        help=(
            "Pass no-defense-fedavg=True to the client.  "
            "Clients will send raw weights instead of gradient updates.  "
            "Use this for pure FedAvg experiments."
        ),
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Override the results output directory.",
    )
    parser.add_argument(
        "--gpu-resources", type=float, default=0.0,
        help="GPU resources to allocate per client (e.g. 0.2 for 5 clients/GPU, default: 0.0)",
    )
    return parser


def run_experiment(args: argparse.Namespace, make_strategy_fn, strategy_name: str) -> None:
    """Run a Flower simulation experiment.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments from `build_arg_parser`.
    make_strategy_fn : callable
        Signature: ``(args, initial_params, num_rounds, fit_fn, eval_fn) -> Strategy``
    strategy_name : str
        Short name used for the results directory (e.g. ``"fedavg"``).
    """
    num_rounds = args.rounds
    dataset_name = args.dataset
    num_clients = args.num_clients
    iid_level = args.iid_level
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    attacks_label = args.attack_patterns.replace(",", "-") if args.attack_patterns else "no-attack"
    
    # Construct a descriptive output directory
    default_out_dir = (
        f"results/{dataset_name}/{attacks_label}/iid{iid_level}/"
        f"{strategy_name}/{timestamp}"
    )
    out_dir = args.out_dir or default_out_dir
    args.out_dir = out_dir  # update args so strategies get the right path

    non_iid, dirichlet_alpha = iid_level_to_params(iid_level)

    print(f"\n{'='*60}")
    print(f"  Strategy      : {strategy_name.upper()}")
    print(f"  Dataset       : {dataset_name}")
    print(f"  Rounds        : {num_rounds}")
    print(f"  Clients       : {num_clients}")
    print(f"  IID level     : {iid_level} (non_iid={non_iid}, alpha={dirichlet_alpha})")
    print(f"  Malicious prob: {args.malicious_prob}")
    print(f"  Attack patterns: {args.attack_patterns or 'none'}")
    print(f"  Output dir    : {out_dir}")
    print(f"{'='*60}\n")

    # Build initial model parameters
    net = load_model(dataset_name)
    ndarrays = get_weights(net)
    initial_params = ndarrays_to_parameters(ndarrays)

    # Build metric aggregation functions
    fit_fn = build_fit_metrics_fn(num_rounds, out_dir)
    eval_fn = build_eval_metrics_fn(num_rounds, out_dir)

    # Build strategy via caller-supplied factory
    strategy = make_strategy_fn(args, initial_params, num_rounds, fit_fn, eval_fn)

    # Run config for the client app
    attack_patterns_str = args.attack_patterns if args.attack_patterns else ""

    run_config = {
        "num-server-rounds": num_rounds,
        "local-epochs": args.local_epochs,
        "dataset": dataset_name,
        "total-max-samples": args.total_max_samples,
        "non-iid": non_iid,
        "dirichlet-alpha": dirichlet_alpha,
        "iid-level": iid_level,
        "malicious-probability": args.malicious_prob,
        "attack-patterns": attack_patterns_str,
        "update-scaling-factor": args.update_scaling_factor,
        "no-defense-fedavg": args.no_defense_fedavg,
        # FLARE-specific keys (not used by baselines but client_app reads them)
        "fraction-fit": args.fraction_fit,
        "fraction-evaluate": args.fraction_evaluate,
        "base-reliability-threshold": 0.5,
        "alpha": 0.7,
        "beta": 0.6,
        "anomaly_threshold": 3.6,
        "penalty_severity": 2,
        "gamma": 0.3,
        "delta": 0.4,
        "recovery": 0.05,
        "decay": 0.15,
        "late-training-threshold": 0.6,
        "out-dir": out_dir,
    }

    def server_fn(context):
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=num_rounds),
        )

    server_app = ServerApp(server_fn=server_fn)
 
    # Create run and set config
    run_id = generate_rand_int_from_bytes(RUN_ID_NUM_BYTES)
    run = Run.create_empty(run_id=run_id)
    run.override_config = run_config

    run_simulation(
        num_supernodes=num_clients,
        exit_event=EventType.PYTHON_API_RUN_SIMULATION_LEAVE,
        server_app=server_app,
        client_app=client_app,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": args.gpu_resources}},
        verbose_logging=False,
        run=run,
        app_dir=".",
        is_app=True,
        server_app_run_config=run_config,
    )

    print(f"\n✓ Experiment complete.  Results saved to: {os.path.abspath(out_dir)}/\n")
