![Maturity level-0] (https://img.shields.io/badge/Maturity%20Level-ML--0-red)


# Feature Selection with Stochastic Gates (STG) for PopPK Covariate Search
This is a supporting code for "Stochastic Gates for Covariate Selection in Population Pharmacokinetics Modelling" publication. 
**This project is a modified version of the excellent STG work by Run Opti:**
- [Original Project Page](https://runopti.github.io/stg/)
- [2020 ICML Paper](https://proceedings.icml.cc/static/paper_files/icml/2020/5085-Paper.pdf)

---

## Installation

```bash 
pip install .
```

Run the above command in your preferred (activated) virtual environment or using your package manager of choice.

**Note on BorutaShap**
To install BorutaShap from  <https://github.com/Ekeany/Boruta-Shap> (for XGBoost experiments) in the current environment  
folow  <https://github.com/Ekeany/Boruta-Shap/pull/139>
## Project Structure

```
data/
  <scenario>/
    ├── sim_df.csv         # Simulated PK curves per scenario
    ├── col_types.json     # Covariate column types (categorical, continuous)
    ├── train.csv          # Training covariates and ETAs per patient
    └── test.csv           # Test covariates and ETAs per patient

notebooks/
  ├── XGBoost_SHAP_analysis.ipynb  # Functions and plotting for SHAP analysis (reproduce paper figures)
  └── plotting.ipynb               # General plotting utilities for results

scripts/
  ├── find_hyperparameters.py      # Optuna hyperparameter search
  ├── run_train.py                 # Train/test split experiments
  └── run_lambdas_CV.py            # Cross-validation across Lambda grid

src/stg_PopPK/
  ├── models.py       # Modified core STG models
  ├── utils.py        # Utility functions from original STG repo (unchanged)
  ├── stg.py          # Modified STG class
  └── main_utils.py   # Helpers for interacting with STG class (training/validation)

pyproject.toml         # Build and installation metadata
```

## Usage

### 1. Hyperparameter Optimization (Optuna)

Run Optuna trials for any scenario (results saved under `data/<SCENARIO>/optuna_output`):

```bash
python find_hyperparameters.py --scenario reference
```

**Available scenarios**: `reference`, `high_iiv`, `linearly_dependent`, `low_frequency`, `pop_100`, `xor`

### 2. Cross-Validation Experiments

Performs grid search over λ (lambda) — hardcoded values — and saves results to `data/output_CV/lambda/results.json`:

```bash
python run_lambdas_CV.py --scenario reference
```

(Optional) For a custom initial "warm start" for the gates, add `--warm-start [LIST]` (as a list of initial values):

```bash
python run_lambdas_CV.py --scenario reference --warm-start 0.5,0.4,0.1
```

### 3. Final Train/Test Split

Runs final training and testing, saving output to `data/output_CV/output_train_test/lambda/results.json`:

```bash
python run_train.py --scenario reference --lam 0.4
```

With an optional warm start:

```bash
python run_train.py --scenario reference --lam 0.4 --warm-start 0.5,0.4,0.1
```

---

**Notes**

* All scenario names are case-sensitive; use only those listed above.
* Datasets included (in `data/`) correspond to synthetic scenarios described in the original paper: *"Stochastic Gates for Covariate Selection in Population Pharmacokinetics Modelling"*.
* Code structure is organized for easy extension and experiment management.

## License

This repository is provided under the Apache 2.0 License.
See the LICENSE file for details.
