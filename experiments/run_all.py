#!/usr/bin/env python3
"""
experiments/run_all.py
----------------------

Usage:
    python -m experiments.run_all --rounds 5 --dataset mnist --num-clients 10
"""
import argparse
import subprocess
import sys
import os

# Global sweep parameters
BASELINES = ["fedavg", "krum", "multikrum", "trimmed_mean", "fltrust", "flame", "brea", "repunet", "flare"]
DATASETS = ["mnist", "cifar10", "svhn"]
IID_LEVELS = [1.0, 0.7, 0.5, 0.3, 0.1]
ATTACKS = ["label-flipping", "random-update", "update-scaling", "alie", "statistical-mimicry", "adaptive-attack"]

def orchestrate_base_experiment():
    """Parses arguments and runs the global experimental validation sweep."""
    parser = argparse.ArgumentParser(description="Global FL baseline validation sweep")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--malicious-prob", type=float, default=0.1)
    parser.add_argument("--total-max-samples", type=int, default=100)
    parser.add_argument("--gpu-resources", type=float, default=0.0)
    
    args, unknown = parser.parse_known_args()

    print(f"\n{'='*80}")
    print(f"  GLOBAL VALIDATION SWEEP: {len(BASELINES)} Baselines | {len(DATASETS)} Datasets | {len(ATTACKS)} Attacks")
    print(f"  Configuration: {args.num_clients} Clients | {args.rounds} Rounds | {args.malicious_prob} Malicious")
    print(f"{'='*80}\n")

    for ds in DATASETS:
        for iid in IID_LEVELS:
            for atk in ATTACKS:
                print(f"\n>>> SCENARIO: Dataset={ds} | IID={iid} | Attack={atk}")
                print("-" * 60)
                
                for strategy in BASELINES:
                    print(f"RUNNING: {strategy}...")
                    
                    cmd = [
                        sys.executable, "-m", f"experiments.run_{strategy}",
                        "--rounds", str(args.rounds),
                        "--dataset", ds,
                        "--num-clients", str(args.num_clients),
                        "--local-epochs", str(args.local_epochs),
                        "--iid-level", str(iid),
                        "--malicious-prob", str(args.malicious_prob),
                        "--total-max-samples", str(args.total_max_samples),
                        "--attack-patterns", atk,
                        "--gpu-resources", str(args.gpu_resources)
                    ]
                    
                    if strategy in ["krum", "multikrum"]:
                        if "--num-byzantine" not in unknown:
                            cmd += ["--num-byzantine", str(max(1, int(args.num_clients * args.malicious_prob)))]
                    
                    if strategy == "multikrum":
                        if "--top-k" not in unknown:
                            cmd += ["--top-k", str(max(1, int(args.num_clients * 0.6)))]
                    
                    if strategy == "trimmed_mean":
                        if "--trim-fraction" not in unknown:
                            cmd += ["--trim-fraction", str(min(0.4, args.malicious_prob + 0.1))]

                    cmd.extend(unknown)

                    try:
                        subprocess.run(cmd, check=True)
                    except Exception as e:
                        print(f"✗ {strategy} failed in scenario: {ds}/{iid}/{atk}\n")
                        continue

    print(f"\n{'='*80}")
    print("  Global experimental sweep complete.")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    orchestrate_base_experiment()
