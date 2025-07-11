import pandas as pd
import numpy as np
import json
from stg_PopPK.main_utils import run_train
import argparse
import os
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_PATH / "data"


def read_data(scenario):
    csv_data_train = pd.read_csv(DATA_PATH / scenario / "train.csv")
    csv_data_test = pd.read_csv(DATA_PATH / scenario / "test.csv")
    with open(DATA_PATH / scenario / "col_types.json") as f:
        col_types = json.load(f)
    return csv_data_train, csv_data_test, col_types


best_model_params = dict(
    optimizer="Adam",
    activation="tanh",
    learning_rate=0.001,
    weight_decay=0.001,
    hidden_dims=[20, 20],
    dropout=0.5,
)


def run_train_test(lam, scenario, warm_start=None, folder_suffix=None):
    data_train, data_test, col_types = read_data(scenario)
    X_train = data_train[[*col_types["cat"], *col_types["con"]]]
    y_train = data_train[col_types["eta"]]

    X_test = data_test[[*col_types["cat"], *col_types["con"]]]
    y_test = data_test[col_types["eta"]]

    if warm_start is not None:
        warm_start = np.array(warm_start)
        assert len(warm_start) == X_test.shape[1]

    output_folder = f"{DATA_PATH}/{scenario}/output_train_test_{folder_suffix}/"
    os.makedirs(output_folder, exist_ok=True)
    nepoch = 1000
    totepoch = 10000

    output_folder_lam = os.path.join(output_folder, f"{lam}")
    os.makedirs(output_folder_lam, exist_ok=True)
    columns = X_train.columns

    out_dict = run_train(
        trainX=X_train,
        trainY=y_train,
        testX=X_test,
        testY=y_test,
        model_params=best_model_params,
        max_size=1500,
        lam=lam,
        stop_epoch=nepoch,
        totepoch=totepoch,
        columns=columns,
        catcols=col_types["cat"],
        do_shap=False,
        save_mus=True,
        save_loss=True,
        warm_start=warm_start,
    )

    outname = os.path.join(output_folder_lam, "results.json")
    with open(outname, "w") as fp:
        json.dump(out_dict, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=["reference", "high_iiv", "linearly_dependent", "low_frequency", "pop_100", "xor"],
        required=True,
    )
    parser.add_argument("--lam", type=float, required=True)
    parser.add_argument("--warm-start", nargs="*", type=float, default=None)
    parser.add_argument("--folder-suffix", type=str, default=None)
    args = parser.parse_args()

    lam = args.lam
    scenario = args.scenario
    run_train_test(lam, scenario, args.warm_start, args.folder_suffix)
