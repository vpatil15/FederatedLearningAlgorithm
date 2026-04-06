# CSEN317 Final Project — Federated Learning Experiments

This repository contains experiments comparing centralized (baseline) training and a simulated Federated Averaging (FedAvg) setup on FashionMNIST and CIFAR-10 using compact convolutional neural networks. The code is implemented in a single script: `csen317_finalproject.py`.

## Key features

- Baseline centralized training with multiple runs and plotting of mean/variance across runs.
- Simulated FedAvg with non-IID client splits using Dirichlet distributions, client local training, aggregation, early stopping, and result visualization.
- Support for both FashionMNIST (grayscale, 1 channel) and CIFAR-10 (RGB, 3 channel).
- Reproducible experiments via fixed seeds and run controls.

## Requirements

- macOS (tested)
- Python 3.8+
- PyTorch (tested with torch >=1.8)
- torchvision
- numpy
- matplotlib

A minimal requirements list (example):

```
torch
torchvision
numpy
matplotlib
```

Install in a virtual environment (zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision numpy matplotlib
```

(If using CUDA, install the appropriate CUDA-enabled PyTorch build; see https://pytorch.org.)

## Files

- `csen317_finalproject.py` — Main script. Downloads datasets, defines models, runs baseline and FedAvg experiments, and produces plots and summary statistics.
- `data/` — (created by torchvision) stores downloaded FashionMNIST and CIFAR-10 datasets.

## How to run

1. Activate your Python virtual environment (see above).
2. From the project directory run:

```bash
python3 csen317_finalproject.py
```

The script will automatically download datasets (if not present), run baseline experiments, run FedAvg simulations, and show plots. Several print statements provide progress and final statistics.

Notes:
- The script uses `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`. For GPU runs, ensure CUDA-enabled PyTorch is installed and a compatible GPU is available.
- Experiments may be time-consuming; reduce `runs`, `epochs`, `rounds`, or `local_epochs` in the script call sites for faster iteration.

## Configuration

Edit the parameters when the functions are invoked near the bottom of `csen317_finalproject.py`:

- Baseline: `runs`, `epochs`, `batch_size`, `lr`, `device`
- FedAvg: `num_clients`, `runs`, `rounds`, `local_epochs`, `batch_size`, `lr`, `patience`, `device`

Example (reduce runtime for quick tests):
```
runs=1
epochs=2
rounds=3
local_epochs=1
```

## Output

- Console logs: per-run and per-round accuracy and loss summaries.
- Plots: Matplotlib windows for training/test curves and comparisons (mean ± std shading).
- Final printed comparisons: mean and variance of final accuracies across runs.

## Reproducibility

The script sets seeds via `set_seed(seed)` and uses deterministic splits for reproducibility where possible. To reproduce an experiment, use the same run seeds and parameter settings.

## Troubleshooting

- Out of memory on GPU: lower `batch_size` or move to CPU by forcing `device='cpu'`.
- Slow downloads: datasets are downloaded via torchvision to `./data/` — try again with a stable connection.
- Missing dependency errors: ensure the virtual environment is activated and packages installed.

## Extending the code

- Add more clients or different non-IID partition strategies by editing `uneven_dirichlet_split` or replacing it.
- Swap the model constructors `SmallCNN_FashionMNIST` / `SmallCNN_CIFAR` for other architectures.
- Replace `optim.SGD` with other optimizers or schedulers.


