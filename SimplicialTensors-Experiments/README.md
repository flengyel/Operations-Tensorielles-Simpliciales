# SimplicialTensors Experiments

This folder contains lightweight experiments that depend on the main `SimplicialTensors` package.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you do not have access to `torch`/`torchvision`, the scripts will fall back to synthetic data when possible.

## Running MNIST A/B

```bash
python experiments/mnist_ab.py --epochs 3 --lambda1 1e-3
```

The script runs two jobs in sequence:

1. Incidence boundary penalty.
2. Principal boundary penalty (zero coboundary).

Metrics for each configuration are printed to stdout. Use the `--help` flag for additional options.
