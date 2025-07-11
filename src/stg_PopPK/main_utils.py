import os
import json
from itertools import product
from typing import Optional, Union, List, Tuple, Dict, Any, Generator
import numpy as np
import torch
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
import shap
from stg_PopPK.stg import STG


def scale_data(
    x_data: np.ndarray,
    y_data: np.ndarray,
    sc1: Optional[StandardScaler] = None,
    sc2: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """
    Scales X and Y data using StandardScaler, optionally reusing provided scalers.
    Returns scaled arrays and used scalers.
    """
    if sc1 is None:
        sc1 = StandardScaler()
        x_scalled = sc1.fit_transform(x_data)
    else:
        x_scalled = sc1.transform(x_data)
    if sc2 is None:
        sc2 = StandardScaler(with_mean=True)
        y_scalled = sc2.fit_transform(y_data)
    else:
        y_scalled = sc2.transform(y_data)

    return x_scalled, y_scalled, sc1, sc2


def get_feature_importance(model: Any, columns: List[str], mode: str = "raw") -> Dict[str, float]:
    """
    Extracts and names feature importance scores using the STG gate values.
    """
    probs = model.get_gates(mode=mode)
    importances = {cname: float(pr) for cname, pr in zip(columns, probs)}
    return importances


def cross_valid_data(
    X_data: pd.DataFrame,
    y_data: Union[pd.Series, pd.DataFrame, np.ndarray],
    ids: pd.Series,
    studyids: pd.Series,
    random_state: int = 0,
    n_splits: int = 5,
    n_bins: int = 10,
) -> Generator[
    Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series], None, None
]:
    """
    Stratified K-Fold cross-validation by quantile-binned target and study ID.
    Yields train/test splits and their indices.
    """
    # Bin the continuous target variable for stratification
    y_binned = pd.qcut(y_data.squeeze(), q=n_bins, duplicates="drop")  # Quantile-based bins
    y_binned = y_binned.astype(str)  # Ensure it's categorical/object for stratification
    stratify_col = y_binned.astype(str) + studyids.astype(str)
    sgkf = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)
    sgkf.get_n_splits(ids, studyids)
    for i, (train_ids_indx, test_ids_indx) in enumerate(sgkf.split(ids, stratify_col)):
        trainX = X_data.iloc[train_ids_indx]
        trainY = y_data.iloc[train_ids_indx]
        ids_train = ids.iloc[train_ids_indx]
        testX = X_data.iloc[test_ids_indx]
        testY = y_data.iloc[test_ids_indx]
        ids_test = ids.iloc[test_ids_indx]
        yield trainX, trainY, ids_train, testX, testY, ids_test


def get_first_fold(
    X_data: pd.DataFrame,
    y_data: Union[pd.Series, pd.DataFrame, np.ndarray],
    ids: pd.Series,
    studyids: pd.Series,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """
    Returns the data for the first cross-validation fold.
    """
    for data in cross_valid_data(X_data, y_data, ids, studyids, random_state=random_state):
        return data


def get_refval_shap(columns, cat_columns, X_shap):
    refvals = []
    for i, col in enumerate(columns):
        if col in cat_columns:
            refval = min(X_shap[:, i])
        else:
            refval = np.median(X_shap[:, i])
        refvals.append(refval)
    return np.array(refvals)


def make_syn_pop(columns, cat_columns, X_shap, refvals):
    population = []
    col_index = []
    for i, col in enumerate(columns):

        if col in cat_columns:
            pat = refvals.copy()
            pat[i] = max(X_shap[:, i])
            population.append(pat)
            col_index.append(col)
        else:

            for p in [5, 95]:
                pat = refvals.copy()
                pat[i] = np.percentile(X_shap[:, i], p)
                population.append(pat)
                col_index.append(f"{col} {p}")
    return np.array(population), col_index


def get_shap_vals(stg_model, sc1, sc2, trainX_sc, testX_sc, columns, catcols):
    if testX_sc is not None:
        full_train = np.concatenate([trainX_sc, testX_sc])
    else:
        full_train = trainX_sc
    refvals = get_refval_shap(columns, catcols, full_train)
    synX, col_index = make_syn_pop(columns, catcols, full_train, refvals)
    stg_model.model.eval()
    shap_output = {}
    reference_values = np.array(refvals)[None, ...]
    explainer = shap.DeepExplainer(
        stg_model.model, torch.from_numpy(reference_values).float().to(stg_model.device)
    )
    for X_shap, name in zip([testX_sc, full_train, synX], ["test", "full"]):
        if X_shap is None:
            continue
        feed_dict_val = torch.from_numpy(X_shap).float().to(stg_model.device)
        shap_values = explainer.shap_values(feed_dict_val)
        # put back in original space but subtract the mean value
        shap_values_org = sc2.inverse_transform(shap_values.squeeze()) - sc2.mean_
        org_X = sc1.inverse_transform(X_shap)
        X_shap_df_aux = pd.DataFrame(data=org_X, columns=columns)
        if name == "synthetic":
            X_shap_df_aux.index = col_index
        shaps_df_aux = pd.DataFrame(data=shap_values_org, columns=columns)
        shap_output[name] = {"data": X_shap_df_aux, "shapv": shaps_df_aux}
    return shap_output


def run_train(
    *,
    trainX: pd.DataFrame,
    trainY: Union[pd.Series, np.ndarray],
    testX: Optional[pd.DataFrame],
    testY: Optional[Union[pd.Series, np.ndarray]],
    model_params: dict,
    max_size: int,
    lam: float,
    stop_epoch: int,
    totepoch: int,
    columns: List[str],
    catcols: List[str],
    do_shap: bool = False,
    save_mus: bool = False,
    save_loss: bool = False,
    warm_start: Optional[np.ndarray] = None,
    shaps_data: str = "test",
    shaps_type: str = "full",
) -> dict:
    """
    Scales data, fits STG, collects feature importance and SHAP if requested, and logs loss/weights if specified.
    """
    # scale data
    trainX_sc, trainY_sc, sc1, sc2 = scale_data(trainX, trainY, sc1=None, sc2=None)
    if testX is not None:
        testX_sc, testY_sc, sc1, sc2 = scale_data(testX, testY, sc1=sc1, sc2=sc2)
    else:
        testX_sc = testY_sc = None

    stg_model = STG(
        device="cuda",
        input_dim=trainX.shape[1],
        output_dim=1,
        **model_params,
        batch_size=max_size,
        sigma=0.5,
        lam=lam,
        save_best=True,
        save_loss=save_loss,
        save_mus=save_mus,
        warm_start=warm_start,
    )

    converged = stg_model.train(
        trainX_sc,
        trainY_sc,
        totepoch,
        testX_sc,
        testY_sc,
        stop_epoch=stop_epoch,
        shuffle=True,
        print_interval=1000,
    )
    if stg_model.best_checkpoint is not None:
        stg_model.model.load_state_dict(stg_model.best_checkpoint)

    feat_import_raw = get_feature_importance(stg_model, columns, mode="prob")
    logger.info(f"lambda: {lam} , val mse best: {stg_model.val_mse_best}")
    logger.info(f"feat_importance:{feat_import_raw}")

    good_fit = bool(stg_model.val_mse_mean > stg_model.val_mse_best)
    if do_shap:
        shap_output = get_shap_vals(stg_model, sc1, sc2, trainX_sc, testX_sc, columns, catcols)
    else:
        shap_output = {}

    if save_loss:
        train_mse_arr = np.array(stg_model.train_loss_arr).astype(np.float32).tolist()
        valid_mse_arr = np.array(stg_model.valid_loss_arr).astype(np.float32).tolist()
        train_r2_arr = np.array(stg_model.train_r2_arr).astype(np.float32).tolist()
        valid_r2_arr = np.array(stg_model.valid_r2_arr).astype(np.float32).tolist()
    else:
        train_mse_arr = valid_mse_arr = train_r2_arr = valid_r2_arr = []

    if save_mus:
        mus = np.array(stg_model.mus_arr).astype(np.float32).tolist()
    else:
        mus = []

    out_dict = dict(
        r2_train=float(stg_model.train_r2_best),
        mse_train=float(stg_model.train_mse_best),
        r2_valid=float(stg_model.val_r2_best),
        mse_valid=float(stg_model.val_mse_best),
        S=float(stg_model.Sbest),
        feat_import_raw=feat_import_raw,
        good_fit=good_fit,
        mus=mus,
        train_mse_arr=train_mse_arr,
        valid_mse_arr=valid_mse_arr,
        train_r2_arr=train_r2_arr,
        valid_r2_arr=valid_r2_arr,
        shaps=shap_output,
    )
    return out_dict


def run_cv(
    model_params,
    lam,
    X_data_eta,
    y_data_eta,
    ids_eta,
    studyids,
    catcols,
    stopepochs,
    totepoch,
    output_folder=None,
    save_mus=False,
    save_loss=False,
    do_shap=False,
    max_size=1024,
    warm_start=None,
):
    columns = X_data_eta.columns
    for fold, (trainX, trainY, ids_train, testX, testY, ids_test) in enumerate(
        cross_valid_data(X_data_eta, y_data_eta, ids_eta, studyids)
    ):
        logger.info(f"fold: {fold}, train size: {trainX.shape}, test_size: {testX.shape}")
        out_dict = run_train(
            trainX=trainX,
            trainY=trainY,
            testX=testX,
            testY=testY,
            model_params=model_params,
            max_size=max_size,
            lam=lam,
            stop_epoch=stopepochs,
            totepoch=totepoch,
            columns=columns,
            catcols=catcols,
            do_shap=do_shap,
            save_mus=save_mus,
            save_loss=save_loss,
            warm_start=warm_start,
        )

        outname = os.path.join(output_folder, f"results_{fold}.json")
        shap_output = out_dict.pop("shaps")
        with open(outname, "w") as fp:
            json.dump(out_dict, fp)
        for name in shap_output.keys():
            X_test_shap = shap_output[name]["data"]
            shaps_df = shap_output[name]["shapv"]
            X_test_shap.to_csv(os.path.join(output_folder, f"shap_X_{name}_{fold}.csv"))
            shaps_df.to_csv(os.path.join(output_folder, f"shap_vals_{name}_{fold}.csv"))


def objective(
    trial: Any,
    trainX: pd.DataFrame,
    trainY: Union[pd.Series, np.ndarray],
    testX: pd.DataFrame,
    testY: Union[pd.Series, np.ndarray],
    stop_epoch: int,
    tot_epoch: int,
    warm_start: Optional[np.ndarray] = None,
) -> float:
    """
    Optuna optimization objective function for tuning STG hyperparameters.
    """
    optimizer_name = "Adam"
    save_mus = False
    save_loss = False
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])
    lam = trial.suggest_categorical("lam", [0.05, 0.1, 0.5, 1, 2])
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    n_layers = trial.suggest_int("n_layers", 1, 2)
    weight_decay = 1e-3
    hidden_units = [10, 20, 50, 100]
    for i in range(n_layers):
        out_features = trial.suggest_categorical("n_units_l{}".format(i), [10, 20, 50, 100])
        hidden_units.append(out_features)
    dropout = trial.suggest_categorical("dropout", [False, 0.5])
    max_size = min(1024, trainX.shape[0])
    # scale data
    trainX_sc, trainY_sc, sc1, sc2 = scale_data(trainX, trainY, sc1=None, sc2=None)
    testX_sc, testY_sc, sc1, sc2 = scale_data(testX, testY, sc1=sc1, sc2=sc2)

    model = STG(
        device="cuda",
        input_dim=trainX.shape[1],
        output_dim=1,
        hidden_dims=hidden_units,
        activation=activation,
        weight_decay=weight_decay,
        dropout=dropout,
        optimizer=optimizer_name,
        learning_rate=lr,
        batch_size=max_size,
        sigma=0.5,
        lam=lam,
        save_best=True,
        save_loss=save_loss,
        save_mus=save_mus,
        warm_start=warm_start,
    )
    model.train(
        trainX_sc,
        trainY_sc,
        tot_epoch,
        testX_sc,
        testY_sc,
        stop_epoch=stop_epoch,
        shuffle=True,
        print_interval=1000,
    )
    return model.val_r2_best
