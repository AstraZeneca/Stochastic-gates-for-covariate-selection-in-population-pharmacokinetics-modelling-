import optuna
import numpy as np
import pandas as pd
import json
import torch

from stg_PopPK.main_utils import objective, get_first_fold
from optuna.trial import TrialState
import argparse
import os
import joblib
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_PATH / "data"


def read_data(scenario):
    csv_data = pd.read_csv(DATA_PATH / scenario / "train.csv")
    with open(DATA_PATH / scenario / "col_types.json") as f:
        col_types = json.load(f)
    return csv_data, col_types


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=["reference", "high_iiv", "linearly_dependent", "low_frequency", "pop_100", "xor"],
        required=True,
    )
    args = parser.parse_args()
    scenario = args.scenario
    output_folder = f"{DATA_PATH}/{scenario}/optuna_output"
    os.makedirs("optuna_output", exist_ok=True)
    csv_data, col_types = read_data(scenario)

    X_data = csv_data[[*col_types["cat"], *col_types["con"]]]
    y_data = csv_data[col_types["eta"]]
    ids = csv_data["ID"]
    studyids = csv_data["STUDYID"]

    totepoch = 10000
    stop_epoch = 500
    n_trials = 50

    trainX, trainY, ids_train, validX, validY, ids_valid = get_first_fold(
        X_data, y_data, ids, studyids, random_state=0
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, trainX, trainY, validX, validY, stop_epoch, totepoch),
        n_trials=n_trials,
        timeout=10000,
    )
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print(" Number of finished trials: ", len(study.trials))
    print(" Number of pruned trials: ", len(pruned_trials))
    print(" Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    trial = study.best_trial
    print(" Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print(" {}: {}".format(key, value))

    hidden_dims = []
    for i in range(trial.params["n_layers"]):
        hidden_dims.append(trial.params[f"n_units_l{i}"])
    best_model_params = dict(
        optimizer="Adam",
        activation=trial.params["activation"],
        learning_rate=trial.params["lr"],
        hidden_dims=hidden_dims,
        lam=trial.params["lam"],
    )
    out_folder = f"{DATA_PATH}/{scenario}/optuna_output/"
    os.makedirs(out_folder, exist_ok=True)
    joblib.dump(study, f"{out_folder}/optuna_study.pkl")

    with open(f"{out_folder}/best_model_param.json", "w") as f:
        json.dump(best_model_params, f, indent=2)
