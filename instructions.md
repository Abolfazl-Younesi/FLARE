# FLARE Framework Usage Instructions

This document provides detailed instructions on how to run federated learning experiments using the FLARE framework and its optimized baselines.

## 1. Running Individual Strategies
Each strategy (FLARE or baseline) can be run individually using its dedicated script in the `experiments/` directory.

### Commands
- **FLARE**: `python3 -m experiments.run_flare [options]`
- **FedAvg**: `python3 -m experiments.run_fedavg [options]`
- **Krum**: `python3 -m experiments.run_krum [options]`
- **Multi-Krum**: `python3 -m experiments.run_multikrum [options]`
- **Trimmed Mean**: `python3 -m experiments.run_trimmed_mean [options]`
- **FLTrust**: `python3 -m experiments.run_fltrust [options]`
- **RepuNet**: `python3 -m experiments.run_repunet [options]`
- **FLAME**: `python3 -m experiments.run_flame [options]`

## 2. Running Scenario-Based Experiments
To compare **all methods** against a specific attack type on a specific dataset in one go, use the `run_scenario.py` script.

### Example
```bash
python3 -m experiments.run_scenario --dataset mnist --attack-patterns random-update --rounds 20 --iid-level 0.5
```
This will sequentially run every baseline and FLARE, saving the results in organized subdirectories.

## 3. Configuration Parameters & Impact

| Parameter | Default | Impact |
| :--- | :--- | :--- |
| `--rounds` | `50` | Total number of FL rounds. Higher rounds allow for better convergence but increase total runtime. |
| `--num-clients` | `100` | Total number of clients in the simulation. Increasing this increases computation but provides a more realistic scale. |
| `--malicious-prob` | `0.1` | Initial probability of a client being malicious. E.g., `0.2` means ~20% of clients are attackers. |
| `--iid-level` | `1.0` | Controls data heterogeneity. **1.0** is fully IID. **0.3** is strongly non-IID (using Dirichlet distribution with alpha=0.5). |
| `--attack-patterns` | `label-flipping` | The type of Byzantine attack. Supports: `label-flipping`, `random-update`, `update-scaling`, `alie`, `statistical-mimicry`. |
| `--dataset` | `mnist` | Dataset to use: `mnist`, `cifar10`, `svhn`. |
| `--total-max-samples`| `100` | Max number of data samples per client. Useful for quick testing when set low (e.g., `10`). |
| `--gpu-resources`| `0.0` | GPU resources per client. Set to `0.1` for 10 clients/GPU or `1.0` for exclusive access. |

## 4. Understanding Non-IID Levels
The `--iid-level` parameter simplifies the control of data distribution:
- **1.0 (IID)**: Data is shuffled and split uniformly among clients.
- **0.7**: Slight non-IID (alpha=10.0).
- **0.5**: Moderate non-IID (alpha=1.0).
- **0.3**: Strong non-IID (alpha=0.5).
- **0.1**: Extreme non-IID (alpha=0.1).

## 5. Result Structure
All results are saved automatically in `results/` with a descriptive path:
`results/[dataset]/[attack]/iid[level]/[method]/[timestamp]/`

Within each folder:
- `performance/`: Convergence accuracy and loss metrics.
- `robustness/`: Detection metrics (TPR, FPR, Accuracy) for robust strategies.
- `reputation/`: (If applicable) Per-client trust/reputation scores per round.
- `latency/`: Aggregation time CSV.
