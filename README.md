# Adaptive Multi-Dimensional Reputation for Robust Client Reliability in Federated Learning

## 1. Quick Start

```bash
# Clone and install
git clone <repo-url> && cd FLARE-main
pip install -e .
```

---

## 2. Environment Requirements

| Component | Version |
|---|---|
| Python | 3.11 (recommended) |
| PyTorch | 2.8.0 |
| TorchVision | 0.23.0 |
| Flower | 1.20.0 |
| scikit-learn | ≥ 1.3 |
| NumPy | ≥ 1.25 |
| SciPy | ≥ 1.10 |
| pandas | ≥ 2.0 |
| ray | ≥ 2.30.0 |

> **GPU**: Optional but recommended for faster training.  
> **CUDA**: 12.x (tested on NVIDIA PyTorch container `nvcr.io/nvidia/pytorch:25.06-py3`).

---

## 3. Installation

### Option A – pip (local / HPC)

```bash
# 1. Create a virtual environment (Python 3.11 recommended)
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Install the package in editable mode (makes byebye_badclients, baselines, experiments importable)
pip install -e .
```

> **GPU users**: Replace the torch/torchvision lines with the matching CUDA wheel.
> Example for CUDA 12.4:
> ```bash
> pip install torch>=2.4.0 torchvision>=0.19.0 --index-url https://download.pytorch.org/whl/cu124
> ```

> **FLAME clustering**: `scikit-learn >= 1.3` (already in `requirements.txt`) ships `HDBSCAN`.
> If you have an older sklearn, uncomment `hdbscan` in `requirements.txt` and run:
> ```bash
> pip install hdbscan
> ```

## 4. Running the Project – Local Simulation

Flower's simulation mode runs all components (SuperLink, ServerApp, ClientApps) in a single process using Ray.  This is the easiest way to get started.

### 4.1 Available datasets

| Config value | Dataset |
|---|---|
| `'ylecun/mnist'` | MNIST (grayscale 28×28, 10 classes) |
| `'ufldl-stanford/svhn'` | SVHN (colour 32×32, 10 classes) |
| `'uoft-cs/cifar10'` | CIFAR-10 (colour 32×32, 10 classes) |
| `'uoft-cs/cifar100'` | CIFAR-100 (colour 32×32, 100 classes) |

---

## 5. Reproducing the FLARE Experiment

the standalone script:

```bash
python -m experiments.run_flare \
  --rounds 10 --dataset svhn --num-clients 100 \
  --iid-level 1.0 --malicious-prob 0.4 \
  --attack-patterns "statistical-mimicry" \
  --fraction-fit 0.2 --fraction-evaluate 0.8
```

Results are saved to `results/flare/`.

---

## 6. Baseline Experiments

All baseline scripts share a common CLI interface. Results are saved as `.json` files in the `results/` directory, compatible with standard plotting tools. Use `--help` to see all available options.

### 6.1 Common options
## Please check all options to be correct and do not rely on the default. Based on the paper figure caption, enable the required options. Every time we need to put this:
`--local-epochs 5 --total-max-samples -1` with 200 rounds and 100 clients
| Option | Default | Description |
|---|---|---|
| `--rounds N` | 5 | Number of FL rounds |
| `--dataset NAME` | `mnist` | One of: mnist, cifar10, svhn |
| `--num-clients N` | 10 | Number of virtual clients |
| `--local-epochs N` | 5 | Local training epochs |
| `--iid-level LEVEL` | 1.0 | Data heterogeneity (1.0/0.7/0.5/0.3/0.1) |
| `--malicious-prob P` | 0.0 | Fraction of malicious clients |
| `--attack-patterns S` | (none) | Comma-sep attack names |
| `--fraction-fit F` | 0.4 | Fraction trained each round |
| `--total-max-samples N` | 500 | Max samples per client (-1=all) |
| `--out-dir PATH` | `results/<strategy>` | Results directory |

### 6.2 GPU Acceleration
For simulations on machines with NVIDIA GPUs, you can significantly accelerate training using the `--gpu-resources` flag. This uses the Ray backend to manage fractional GPU allocation.

- **`--gpu-resources 0.1`**: Recommended. 10 clients will share the GPU simultaneously. This is often the most efficient for models like ResNet18.
- **`--gpu-resources 1.0`**: Exclusive access. Only one client trains on the GPU at a time.
- **`--gpu-resources 0.0`**: Default. Simulation runs on CPU only.

Some examples Example:
CLEAN IID CIFAR10 with 200 rounds, 100 clients
```bash
python -m experiments.run_flare --num-clients 100 --dataset cifar10 --gpu-resources 0.1 --rounds 200 --local-epochs 5 --total-max-samples -1
```

**FedAvg (standard, no defense)**
```bash
python -m experiments.run_fedavg --rounds 5 --dataset mnist --iid-level 1.0 --rounds 200 
```

**Krum** (Byzantine-robust, picks best single update)
```bash
python -m experiments.run_krum --rounds 5 --dataset mnist --num-byzantine 2 \
  --malicious-prob 0.2 --attack-patterns "label-flipping"
```

**Multi-Krum** (averages top-K by Krum score)
```bash
python -m experiments.run_multikrum --rounds 5 --dataset mnist \
  --num-byzantine 2 --top-k 6 --malicious-prob 0.2
```

**Trimmed Mean** (coordinate-wise, trim extremes)
```bash
python -m experiments.run_trimmed_mean --rounds 5 --dataset cifar10 \
  --trim-fraction 0.2 --malicious-prob 0.3
```

**FLTrust** (cosine-similarity trust scoring)
```bash
python -m experiments.run_fltrust --rounds 5 --dataset mnist \
  --malicious-prob 0.3 --attack-patterns "update-scaling"
```

**FLAME** (HDBSCAN clustering + noise injection)
```bash
python -m experiments.run_flame --rounds 5 --dataset mnist \
  --noise-multiplier 0.001 --malicious-prob 0.3
```

**BREA** (Bayesian iterative-reweighting)
```bash
python -m experiments.run_brea --rounds 5 --dataset mnist \
  --max-iters 5 --malicious-prob 0.3
```

**RepuNet** (EMA reputation-based weighting)
```bash
python -m experiments.run_repunet --rounds 5 --dataset mnist \
  --ema-decay 0.9 --malicious-prob 0.3
```

**FLARE (WeightedFedAvg – full custom strategy)**
```bash
python -m experiments.run_flare --rounds 10 --dataset svhn --num-clients 100 \
  --iid-level 1.0 --malicious-prob 0.4 --attack-patterns "statistical-mimicry"
```

### 6.3 Global Experimental Sweep

For extensive technical benchmarking, the `run_all.py` script implements a comprehensive **Global Validation Sweep**. It automatically orchestrates a nested experimental matrix across all protocols and environmental dimensions.

**Sweep Matrix Dimensions:**
- **Datasets**: `[MNIST, CIFAR-10, SVHN]`
- **Heterogeneity (IID)**: `[1.0, 0.7, 0.5, 0.3, 0.1]` (Dirichlet $\alpha$ spectrum)
- **Adversarial Scenarios**: `[label-flipping, random-update, update-scaling, alie, statistical-mimicry, adaptive-attack]`
- **Protocols**: All 9 supported baselines (FedAvg, Krum, FLARE, etc.)

**Execution Details:**
By default, the sweep is configured for high-fidelity demonstration:
- **Clients**: 100
- **Rounds**: 200
- **Malicious Probability**: 0.1 (configurable)

```bash
# Execute the complete global validation sweep
python -m experiments.run_all --rounds 200 --num-clients 100 --malicious-prob 0.1
```

> [!TIP]
> This sweep produces a massive volume of telemetry across the `results/` directory, which is subsequently synthesized into academic artifacts by the internal visualization suite.

## 7. IID / Non-IID Data Configuration

> **Current status**: The default `pyproject.toml` uses `iid-level = 1.0` (**fully IID**).  
> Previously, the project defaulted to Non-IID (`dirichlet-alpha=0.5`).

### 7.1 The `iid-level` parameter

Use a single `iid-level` value — it automatically selects the right Dirichlet alpha:

| `iid-level` | Distribution | Dirichlet α | Description |
|---|---|---|---|
| **1.0** | Fully IID | N/A | Uniform random split across clients |
| **0.7** | Slightly Non-IID | α = 10.0 | Minor class imbalance |
| **0.5** | Moderately Non-IID | α = 1.0 | Noticeable class skew |
| **0.3** | Strongly Non-IID | α = 0.5 | High class concentration per client |
| **0.1** | Extremely Non-IID | α = 0.1 | Each client holds ~1-2 classes |

### 7.2 Usage

**In `pyproject.toml`:**
```toml
iid-level = 0.3   # Strongly Non-IID
```

**Via experiment scripts:**
```bash
python -m experiments.run_flare --iid-level 0.3 --rounds 5
```

**In Python code:**
```python
from byebye_badclients.task import iid_level_to_params, load_data
non_iid, alpha = iid_level_to_params(0.3)  # → (True, 0.5)
trainloader, testloader = load_data(
    partition_id=0, num_partitions=10,
    dataset="mnist", iid_level=0.3
)
```

---

## 8. Configuration Reference

Full list of `pyproject.toml` config keys:

| Key | Default | Description |
|---|---|---|
| `num-server-rounds` | 1 | Number of FL rounds |
| `fraction-fit` | 0.2 | Fraction of clients sampled for training |
| `fraction-evaluate` | 0.8 | Fraction of clients sampled for evaluation |
| `local-epochs` | 5 | Local training epochs per round |
| `dataset` | `'ufldl-stanford/svhn'` | Dataset identifier |
| `total-max-samples` | 2000 | Max training samples per client (-1=all) |
| **`iid-level`** | **1.0** | **Primary IID/Non-IID knob** |
| `non-iid` | false | Legacy: use Dirichlet non-IID partitioning |
| `dirichlet-alpha` | 0.5 | Legacy: Dirichlet concentration parameter |
| `malicious-probability` | 0.4 | Fraction of clients that are adversarial |
| `attack-patterns` | `'statistical-mimicry'` | Comma-sep attack names |
| `update-scaling-factor` | 2 | Multiplier for update-scaling attack |
| `no-defense-fedavg` | false | If true, uses vanilla FedAvg (no FLARE) |
| `base-reliability-threshold` | 0.5 | FLARE: initial trust threshold |
| `alpha` | 0.7 | FLARE: performance consistency EMA factor |
| `beta` | 0.6 | FLARE: temporal behaviour weight |
| `anomaly_threshold` | 3.6 | FLARE: Mahalanobis distance cutoff |
| `penalty_severity` | 2 | FLARE: anomaly penalty exponent |
| `gamma` | 0.3 | FLARE: threshold–convergence coupling |
| `delta` | 0.4 | FLARE: threshold–anomaly-rate coupling |
| `recovery` | 0.05 | FLARE: reputation recovery rate |
| `decay` | 0.15 | FLARE: reputation decay rate |
| `late-training-threshold` | 0.6 | FLARE: late-training convergence threshold |

---

## 9. Results Structure

The framework produces two tiers of results: raw telemetry from protocol execution and synthesized academic artifacts from the validation suite.

### 9.1 Raw Telemetry
Each experiment writes CSV files to `results/<strategy>/`:
```
results/
├── fedavg/
│   ├── weighted_avg_loss.csv         (fit metrics per round)
│   ├── weighted_avg_accuracy.csv     (eval metrics per round)
│   ├── weighted_avg_f1_score.csv
│   ├── weighted_avg_precision_score.csv
│   ├── weighted_avg_recall_score.csv
│   └── latency/aggregation_time.csv
...
```

### 9.2 Analytical Artifacts
The post-execution verification suite synthesizes the raw data into high-fidelity academic reports in the `results/` and `plots/` directories:

**Telemetry Models (`results/`):**
- `empirical_bounds_{X}pct.csv`: Robustness bounds per malicious fraction.
- `convergence_trajectories.csv`: Asymptotic efficiency over rounds.
- `reputation_dynamics_modeled.csv`: Temporal trust metrics for canonical profiles.
- `attack_success_rates.csv`: Temporal adversarial success probability.

**Diagnostic Plots (`plots/`):**
- `robustness/`: Accuracy heatmaps across attack/dataset manifold.
- `convergence/`: Multi-protocol temporal efficiency comparisons.
- `reputation_evolution_dynamics.pdf`: High-fidelity trust profile tracking.
- `best_method_summary_heatmap.pdf`: Definitive performance mapping across scenarios.

---

## 10. Extending the Framework

### Add a new strategy

1. Create `baselines/my_strategy.py` subclassing `TimingMixin` + `FedAvg`:

```python
from baselines.common import TimingMixin, alie_malicious_network_hook
from flwr.server.strategy import FedAvg

class MyStrategy(TimingMixin, FedAvg):
    def __init__(self, server_rounds=10, out_dir="results/my_strategy", **kwargs):
        FedAvg.__init__(self, **kwargs)
        TimingMixin.__init__(self, server_rounds=server_rounds, out_dir=out_dir)

    def aggregate_fit(self, server_round, results, failures):
        alie_malicious_network_hook(results)
        # ... your aggregation logic ...
```

2. Create `experiments/run_my_strategy.py` following the pattern in `experiments/run_krum.py`.
3. Register it in `baselines/__init__.py`.

### Add a new dataset

Add a new branch in `byebye_badclients/task.py` → `load_data()` and handle `.targets` or label extraction for Dirichlet partitioning.

---

## 11. Troubleshooting

### TensorFlow/Protobuf Version Conflicts
The project uses `flwr < 5.0.0` which can have protobuf version conflicts with recent `tensorflow`.
- **Our Fix**: We have patched `flwr.server.utils.tensorboard` and `flwr.simulation.ray_transport.utils` to make the TensorFlow import optional. If TensorFlow is broken, Flower will still run experiments.
- **Manual Workaround**: If you encounter `ImportError: google.protobuf.runtime_version`, ensure `protobuf` is pinned to `>=4.21.6,<5.0.0`.

### WeightedFedAvg (FLARE) Strategy Crash
If `MinCovDet` (MCD) fails with a `ValueError: array must not contain infs or NaNs`:
- This typically happens when the `support_fraction` is too low for the number of features and samples.
- **Our Fix**: We have added logic to `server_app.py` to automatically calculate the minimum valid `support_fraction` and added jitter for rank-deficient matrices to ensure numerical stability.

### TypeError: run_simulation() got an unexpected keyword argument 'run_config'
Recent versions of Flower simulation API have changed.
- **Our Fix**: Standalone scripts in the `experiments/` directory have been updated to use the internal `_run_simulation` API and properly initialize the `Run` object for context propagation.
