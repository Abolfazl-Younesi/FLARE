#!/usr/bin/env python3
"""
experiments/run_flare.py
-------------------------
FLARE (WeightedFedAvg / ByeBye-BadClients) experiment.

This is the original custom strategy from the FLARE paper, wrapped as a
standalone experiment script for easy comparison with baselines.

Usage:
    python -m experiments.run_flare [options]
    python -m experiments.run_flare --help
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse

from flwr.common import ndarrays_to_parameters, EventType
from flwr.common.typing import Run
from flwr.common.constant import RUN_ID_NUM_BYTES
from flwr.server.superlink.linkstate.utils import generate_rand_int_from_bytes
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation.run_simulation import _run_simulation as run_simulation

from byebye_badclients.server_app import WeightedFedAvg
from byebye_badclients.task import get_weights
from byebye_badclients.util import load_model
from byebye_badclients.client_app import app as client_app
from baselines.common import (
    iid_level_to_params,
    build_fit_metrics_fn,
    build_eval_metrics_fn,
)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="FLARE (WeightedFedAvg) experiment")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "cifar10", "svhn", "cifar100"])
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--fraction-fit", type=float, default=0.4)
    parser.add_argument("--fraction-evaluate", type=float, default=0.5)
    parser.add_argument("--iid-level", type=float, default=1.0,
                        choices=[1.0, 0.7, 0.5, 0.3, 0.1])
    parser.add_argument("--malicious-prob", type=float, default=0.0)
    parser.add_argument("--attack-patterns", type=str, default="")
    parser.add_argument("--update-scaling-factor", type=float, default=2.0)
    parser.add_argument("--total-max-samples", type=int, default=500)
    parser.add_argument("--out-dir", type=str, default="results/flare")
    # FLARE-specific hyperparameters
    parser.add_argument("--base-reliability-threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--anomaly-threshold", type=float, default=3.6)
    parser.add_argument("--penalty-severity", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--delta", type=float, default=0.4)
    parser.add_argument("--recovery", type=float, default=0.05)
    parser.add_argument("--decay", type=float, default=0.15)
    parser.add_argument("--late-training-threshold", type=float, default=0.6)
    parser.add_argument("--gpu-resources", type=float, default=0.0,
                        help="GPU resources to allocate per client (default: 0.0)")
    return parser


def main():
    args = build_arg_parser().parse_args()
    num_rounds = args.rounds
    dataset_name = args.dataset
    num_clients = args.num_clients
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    attacks_label = args.attack_patterns.replace(",", "-") if args.attack_patterns else "no-attack"
    iid_level = args.iid_level
    
    default_out_dir = (
        f"results/{dataset_name}/{attacks_label}/iid{iid_level}/"
        f"flare/{timestamp}"
    )
    out_dir = args.out_dir if args.out_dir != "results/flare" else default_out_dir
    non_iid, dirichlet_alpha = iid_level_to_params(iid_level)

    print(f"\n{'='*60}")
    print(f"  Strategy      : FLARE (WeightedFedAvg / ByeBye-BadClients)")
    print(f"  Dataset       : {dataset_name}")
    print(f"  Rounds        : {num_rounds}")
    print(f"  Clients       : {num_clients}")
    print(f"  IID level     : {iid_level} (non_iid={non_iid}, alpha={dirichlet_alpha})")
    print(f"  Malicious prob: {args.malicious_prob}")
    print(f"  Output dir    : {out_dir}")
    print(f"{'='*60}\n")

    net = load_model(dataset_name)
    ndarrays = get_weights(net)
    initial_params = ndarrays_to_parameters(ndarrays)

    fit_fn = build_fit_metrics_fn(num_rounds, out_dir)
    eval_fn = build_eval_metrics_fn(num_rounds, out_dir)

    strategy = WeightedFedAvg(
        server_rounds=num_rounds,
        out_dir=out_dir,
        base_reliability_threshold=args.base_reliability_threshold,
        alpha=args.alpha,
        beta=args.beta,
        anomaly_threshold=args.anomaly_threshold,
        penalty_severity=args.penalty_severity,
        gamma=args.gamma,
        delta=args.delta,
        recovery=args.recovery,
        decay=args.decay,
        late_training_threshold=args.late_training_threshold,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_available_clients=1,
        min_fit_clients=1,
        min_evaluate_clients=1,
        initial_parameters=initial_params,
        fit_metrics_aggregation_fn=fit_fn,
        evaluate_metrics_aggregation_fn=eval_fn,
        on_fit_config_fn=lambda r: {"send_time": time.time(), "dataset": dataset_name},
        on_evaluate_config_fn=lambda r: {"dataset": dataset_name},
    )

    run_config = {
        "num-server-rounds": num_rounds,
        "local-epochs": args.local_epochs,
        "dataset": dataset_name,
        "total-max-samples": args.total_max_samples,
        "non-iid": non_iid,
        "dirichlet-alpha": dirichlet_alpha,
        "iid-level": iid_level,
        "malicious-probability": args.malicious_prob,
        "attack-patterns": args.attack_patterns,
        "update-scaling-factor": args.update_scaling_factor,
        "no-defense-fedavg": False,
        "fraction-fit": args.fraction_fit,
        "fraction-evaluate": args.fraction_evaluate,
        "base-reliability-threshold": args.base_reliability_threshold,
        "alpha": args.alpha,
        "beta": args.beta,
        "anomaly_threshold": args.anomaly_threshold,
        "penalty_severity": args.penalty_severity,
        "gamma": args.gamma,
        "delta": args.delta,
        "recovery": args.recovery,
        "decay": args.decay,
        "late-training-threshold": args.late_training_threshold,
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


if __name__ == "__main__":
    main()
