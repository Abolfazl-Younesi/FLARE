#!/usr/bin/env python3
"""
experiments/run_scenario.py
--------------------------

Run all baseline methods for a specific dataset and attack type.
This allows targeted comparison of all strategies in a single scenario.

Usage:
    python -m experiments.run_scenario --dataset mnist --attack-patterns label-flipping --rounds 10
"""
import argparse
import subprocess
import sys
import os

# Available baselines and defaults
BASELINES = ["fedavg", "krum", "multikrum", "trimmed_mean", "fltrust", "flame", "brea", "repunet", "flare"]
DATASETS = ["mnist", "cifar10", "svhn"]
ATTACKS = ["label-flipping", "random-update", "update-scaling", "alie", "statistical-mimicry", "adaptive-attack"]

def run_scenario():
    parser = argparse.ArgumentParser(description="Run all FL baselines for a specific scenario")
    parser.add_argument("--rounds", type=int, default=50, help="Number of FL rounds")
    parser.add_argument("--num-clients", type=int, default=100, help="Total number of clients")
    parser.add_argument("--local-epochs", type=int, default=5, help="Local training epochs")
    parser.add_argument("--dataset", type=str, default="mnist", choices=DATASETS, help="Dataset to use")
    parser.add_argument("--attack-patterns", type=str, default="label-flipping", choices=ATTACKS, help="Attack type")
    parser.add_argument("--iid-level", type=float, default=1.0, help="IID level (1.0=IID, <1.0=non-IID)")
    parser.add_argument("--malicious-prob", type=float, default=0.1, help="Probability of a client being malicious")
    parser.add_argument("--total-max-samples", type=int, default=100, help="Max samples per client")
    parser.add_argument("--gpu-resources", type=float, default=0.1, help="GPU resources per client")
    
    args, unknown = parser.parse_known_args()

    print(f"\n{'='*80}")
    print(f"  SCENARIO EXPERIMENT: All Methods | Dataset={args.dataset} | Attack={args.attack_patterns}")
    print(f"  Configuration: {args.num_clients} Clients | {args.rounds} Rounds | IID={args.iid_level} | Malicious={args.malicious_prob}")
    print(f"{'='*80}\n")

    for strategy in BASELINES:
        print(f"\n>>> RUNNING: {strategy}...")
        print("-" * 40)
        
        cmd = [
            sys.executable, "-m", f"experiments.run_{strategy}",
            "--rounds", str(args.rounds),
            "--dataset", args.dataset,
            "--num-clients", str(args.num_clients),
            "--local-epochs", str(args.local_epochs),
            "--iid-level", str(args.iid_level),
            "--malicious-prob", str(args.malicious_prob),
            "--total-max-samples", str(args.total_max_samples),
            "--attack-patterns", args.attack_patterns,
            "--gpu-resources", str(args.gpu_resources)
        ]
        
        # Add baseline-specific defaults if not overridden by unknown args
        if strategy in ["krum", "multikrum"]:
            if "--num-byzantine" not in unknown:
                cmd += ["--num-byzantine", str(max(1, int(args.num_clients * args.malicious_prob)))]
        
        if strategy == "multikrum":
            if "--top-k" not in unknown:
                cmd += ["--top-k", str(max(1, int(args.num_clients * 0.6)))]
        
        if strategy == "trimmed_mean":
            if "--trim-fraction" not in unknown:
                cmd += ["--trim-fraction", str(min(0.4, args.malicious_prob + 0.1))]

        # Pass through any additional unknown arguments
        cmd.extend(unknown)

        try:
            subprocess.run(cmd, check=True)
            print(f"✓ {strategy} completed successfully.")
        except subprocess.CalledProcessError:
            print(f"✗ {strategy} failed.")
        except Exception as e:
            print(f"✗ {strategy} encountered an error: {e}")

    print(f"\n{'='*80}")
    print("  Scenario experiment complete.")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    run_scenario()
