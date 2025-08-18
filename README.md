# OAK — Orthogonal Additive Kernels (cleaned)

I built this cleaned copy of an Orthogonal Additive Kernel (OAK) codebase as the starting point for my MEng investigations into mutual information across kernel components and how OAK compares to standard additive Gaussian processes (AGPs) on toy and UCI-style datasets.

This repository contains the implementation I used to:

- explore per-component contributions (Sobol indices) and convert those to mutual-information style summaries;
- compare OAK against baseline additive and RBF models across toy datasets, measuring predictive performance and component attribution;
- study loss landscapes for OAK and AGP training (details and extended figures are presented in my MEng thesis).

The codebase includes:

- the OAK kernel and constrained kernels for binary / categorical / continuous inputs
- input measures and optional normalising flows for transforming inputs
- a compact model API for training, predicting and extracting per-component contributions
- example scripts and notebooks used in my experiments

## Repository layout (cleaned)

Top-level (inside this CLEANED/ folder):

- `README.md` — this file (research-focused)
- `setup.py`, `requirements.txt` — packaging / dependency manifests
- `LICENSE`, `NOTICE`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`
- `src/` — package source
  - `src/oak/` — core OAK implementation and utilities (kernels, model_utils, plotting, utils, normalising flow)
  - `src/mutual_information/oak_mi.py` — mutual-information helpers used in notebooks
- `examples/uci/` — example scripts used for UCI regression & classification experiments
- `notebooks/` — curated notebooks (mutual-information notebook, UCI example, contraction-rate experiments)
- `results/` — saved outputs and models from example runs
- `tests/` — unit tests copied from the original project

## run the code

I recommend creating a dedicated environment (Python 3.8+). From the repository root:

```powershell
# create and activate a virtual environment (Windows PowerShell)
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# install dependencies
python -m pip install -r CLEANED/requirements.txt

# optional: install package in editable mode
python -m pip install -e CLEANED
```


## Reproduce a quick mutual-information run

I keep a working notebook at `CLEANED/notebooks/oak_mi.ipynb` that shows how to:

1. train an OAK model on a toy dataset;
2. extract per-component predictions and compute Sobol indices;
3. convert component variances to mutual-information estimates (shown in bits);
4. produce per-component plots and tables that I used in write-ups.

To run the UCI regression example:

```powershell
python CLEANED/examples/uci/uci_regression_train.py --dataset_name=autoMPG
python CLEANED/examples/uci/uci_plotting.py --dataset_name=autoMPG
```

The `notebooks/` folder includes an executable notebook version of the autoMPG example and the mutual-information analysis I ran during the project.

## What was analysed and where to find more detail

- Mutual-information experiments: notebooks and the `src/mutual_information/oak_mi.py` code walk through how I mapped normalized Sobol indices to a per-component information quantity using the Gaussian variance formula. That is the starting point I used to compare how much information each additive term carries.

- OAK vs AGP comparisons: examples under `CLEANED/examples/uci/` and the toy notebooks compare predictive performance, decomposition quality, and component stability across seeds.

- Loss landscapes and training behaviour: richer visualisations and analysis of loss landscapes for OAK and AGP (used to diagnose training stability and local minima) are summarised in my MEng thesis; see the thesis for extended figures, captions and discussion.

