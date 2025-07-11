"""
Script that performs grid search over lambda parameter
"""

import pandas as pd
import json
import numpy as np
from stg_PopPK.main_utils import run_cv, run_train
import argparse
import os
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_PATH / "data"


def read_data(scenario):
    csv_data = pd.read_csv(DATA_PATH / scenario / "train.csv")
    with open(DATA_PATH / scenario / "col_types.json") as f:
        col_types = json.load(f)
    return csv_data, col_types


best_model_params = dict(
    optimizer="Adam",
    activation="tanh",
    learning_rate=0.001,
    weight_decay=0.001,
    hidden_dims=[20, 20],
    dropout=0.5,
)


def run_lambda(lams, scenario, warm_start):
    data, col_types = read_data(scenario)
    X_data = data[[*col_types["cat"], *col_types["con"]]]
    y_data = data[col_types["eta"]]
    ids = data["ID"]
    studyids = data["STUDYID"]
    output_folder = f"{DATA_PATH}/{scenario}/output_CV"
    os.makedirs(output_folder, exist_ok=True)
    nepoch = 1000
    totepoch = 200000
    if warm_start is not None:
        warm_start = np.array(warm_start)
        assert len(warm_start) == X_data.shape[1]
    for lam in lams:
        print("#####################")
        print(f"LAMBDA {lam}")
        print("#####################")
        output_folder_lam = os.path.join(output_folder, f"{lam}")
        os.makedirs(output_folder_lam, exist_ok=True)
        run_cv(
            best_model_params,
            lam,
            X_data,
            y_data,
            ids,
            studyids,
            col_types["cat"],
            nepoch,
            totepoch,
            output_folder=output_folder_lam,
            do_shap=False,
            save_mus=True,
            save_loss=True,
            max_size=1500,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=["reference", "high_iiv", "linearly_dependent", "low_frequency", "pop_100", "xor"],
        required=True,
    )
    parser.add_argument("--warm-start", nargs="*", type=float, default=None)

    args = parser.parse_args()

    lams = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5]
    run_lambda(lams, args.scenario, args.warm_start)
