'''
This script runs a logistic regression model on the dataset. It includes data preprocessing, 
model training, and evaluation of the model's performance using various metrics. 
Codes from: build eventt table.ipynb, build feature table.ipynb, pipeline-LG.ipynb
AI Usage: The codes in this file are assisted by ChatGPT and GitHub Copilot.
'''

# Import necessary libraries
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score,recall_score,f1_score,balanced_accuracy_score,accuracy_score,brier_score_loss,roc_auc_score,average_precision_score,confusion_matrix,classification_report)
from scipy.stats import binomtest
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.compose import ColumnTransformer
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from datetime import datetime as dt
import os

#from google.colab import drive
#drive.mount('/content/drive')

# File Paths
script_path = "final_code/"
data_path = "data/"
output_path = "outputs/"

## Build Event Table for features that will be used ## -------------------------------------------------------------
# Load datasets
vix_table = pd.read_csv(data_path + "vix_table_clean.csv")
oas_table = pd.read_csv(data_path + "hy_oas.csv")
skew_table = pd.read_csv(data_path + "skew_table_clean.csv")
ofr_table = pd.read_csv(data_path + "labels.csv")
nfci_table = pd.read_csv(data_path + "NFCI.csv")
stlf_table = pd.read_csv(data_path + "STLFSI4.csv")

@dataclass
class EventTableBuilder:
    method: str = "level_high"
    lookback: int = 126
    min_periods: int = 63
    q: float = 0.95

    def build(self, data, date_col, value_col, series_name=None):
        series_name = series_name or value_col

        event_obs = data[[date_col, value_col]].copy()
        event_obs = event_obs.rename(columns={date_col: "date", value_col: "value"})

        event_obs["date"] = pd.to_datetime(event_obs["date"])
        event_obs["value"] = pd.to_numeric(event_obs["value"], errors="coerce")
        event_obs = event_obs.sort_values("date").reset_index(drop=True)

        if self.method == "level_high":
            event_obs["metric"] = event_obs["value"]
            event_obs["threshold"] = (
                event_obs["metric"]
                .rolling(self.lookback, min_periods=self.min_periods)
                .quantile(self.q)
                .shift(1)
            )
            event_obs["event_flag"] = event_obs["metric"] > event_obs["threshold"]

        elif self.method == "level_low":
            event_obs["metric"] = event_obs["value"]
            event_obs["threshold"] = (
                event_obs["metric"]
                .rolling(self.lookback, min_periods=self.min_periods)
                .quantile(self.q)
                .shift(1)
            )
            event_obs["event_flag"] = event_obs["metric"] < event_obs["threshold"]

        elif self.method == "delta_pos":
            event_obs["delta"] = event_obs["value"].diff()
            event_obs["metric"] = event_obs["delta"].clip(lower=0)
            event_obs["threshold"] = (
                event_obs["metric"]
                .rolling(self.lookback, min_periods=self.min_periods)
                .quantile(self.q)
                .shift(1)
            )
            event_obs["event_flag"] = event_obs["metric"] > event_obs["threshold"]

        elif self.method == "delta_abs":
            event_obs["delta"] = event_obs["value"].diff()
            event_obs["metric"] = event_obs["delta"].abs()
            event_obs["threshold"] = (
                event_obs["metric"]
                .rolling(self.lookback, min_periods=self.min_periods)
                .quantile(self.q)
                .shift(1)
            )
            event_obs["event_flag"] = event_obs["metric"] > event_obs["threshold"]

        else:
            raise ValueError(
                "method must be one of: level_high, level_low, delta_pos, delta_abs"
            )

        valid = event_obs["metric"].notna() & event_obs["threshold"].notna()
        event_obs.loc[~valid, "event_flag"] = pd.NA

        event_obs["event_entry"] = (
            (event_obs["event_flag"] == True)
            & (event_obs["event_flag"].shift(1) != True)
        )

        event_obs["event_id"] = event_obs["event_entry"].cumsum()
        event_obs.loc[event_obs["event_flag"] != True, "event_id"] = pd.NA

        event_obs["series"] = series_name
        event_obs["method"] = self.method

        event_rows = []

        for event_id, group in event_obs[event_obs["event_flag"] == True].groupby("event_id"):
            peak_idx = group["metric"].idxmax()

            event_rows.append({
                "series": series_name,
                "method": self.method,
                "event_id": int(event_id),
                "start_date": group["date"].min(),
                "end_date": group["date"].max(),
                "duration_obs": len(group),
                "peak_date": event_obs.loc[peak_idx, "date"],
                "peak_value": event_obs.loc[peak_idx, "value"],
                "peak_metric": event_obs.loc[peak_idx, "metric"],
                "threshold_at_peak": event_obs.loc[peak_idx, "threshold"],
            })

        event_table = pd.DataFrame(event_rows)

        return event_obs, event_table
    
level_builder = EventTableBuilder(
    method="level_high",
    lookback=126,
    min_periods=63,
    q=0.95
)

delta_pos_builder = EventTableBuilder(
    method="delta_pos",
    lookback=126,
    min_periods=63,
    q=0.95
)

delta_abs_builder = EventTableBuilder(
    method="delta_abs",
    lookback=126,
    min_periods=63,
    q=0.95
)

# Tables

tables = {
    "vix_table": vix_table,
    "oas_table": oas_table,
    "skew_table": skew_table,
    "ofr_table": ofr_table,
    "nfci_table": nfci_table,
    "stlf_table": stlf_table,
}

for name, table in tables.items():
    print(f"\n{name}")
    print(table.columns.tolist())

# VIX
vix_obs, vix_events = delta_pos_builder.build(vix_table,date_col="date",value_col="Close",series_name="VIX_delta_pos")
vix_level_obs, vix_level_events = level_builder.build(vix_table,date_col="date",value_col="Close",series_name="VIX_level_high")
vix_abs_obs, vix_abs_events = delta_abs_builder.build(vix_table,date_col="date",value_col="Close",series_name="VIX_delta_abs")

# OAS
oas_obs, oas_events = level_builder.build(oas_table, "observation_date", "BAMLH0A0HYM2", series_name="OAS_level_high")

# SKEW
skew_level_obs, skew_level_events = level_builder.build(skew_table, "date", "Close", series_name="SKEW_level_high")

# OFR
ofr_obs, ofr_events = level_builder.build(ofr_table, "Date", "OFR FSI", series_name="OFR_level_high")

# NFCI
nfci_obs, nfci_events = level_builder.build(nfci_table, "observation_date", "NFCI", series_name="NFCI_level_high")

# STLFSI4
stlf_obs, stlf_events = level_builder.build(stlf_table, "observation_date", "STLFSI4", series_name="STLFSI4_level_high")

# OAS delta and abs
oas_delta_obs, oas_delta_events = delta_pos_builder.build(oas_table, "observation_date", "BAMLH0A0HYM2", series_name="OAS_delta_pos")

# OFR delta and abs
ofr_delta_obs, ofr_delta_events = delta_pos_builder.build(ofr_table, "Date", "OFR FSI", series_name="OFR_FSI_delta_pos")

# NFCI delta and abs
skew_delta_obs, skew_delta_events = delta_abs_builder.build(skew_table, "date", "Close", series_name="SKEW_delta_abs")

#out_dir = "/content/drive/MyDrive/557project/event_tables"
out_dir = data_path.copy()
os.makedirs(out_dir, exist_ok=True)

event_tables = {
    "vix_delta_pos_events": vix_events,
    "vix_level_high_events": vix_level_events,
    "vix_delta_abs_events": vix_abs_events,
    "oas_level_high_events": oas_events,
    "skew_level_high_events": skew_level_events,
    "ofr_level_high_events": ofr_events,
    "nfci_level_high_events": nfci_events,
    "stlf_level_high_events": stlf_events,
    "oas_delta_abs_events": oas_delta_events,
    "ofr_delta_abs_events": ofr_delta_events,
    "skew_delta_abs_events": skew_delta_events,
}

# Save event tables to CSV
for name, table in event_tables.items():
    path = os.path.join(out_dir, f"{name}.csv")
    table.to_csv(path, index=False)
    print("saved:", path)


## Build feature tables ## -------------------------------------------------------------
corr_oas_dimension = pd.read_csv(data_path + "dimension_oas_1.csv")
corr_skew_dimension = pd.read_csv(data_path + "corr_dimension_skew_1.csv")
corr_vix_dimension = pd.read_csv(data_path + "corr_dimension_vix_1.csv")
mfdfa_width_df = pd.read_csv(data_path + "mfdfa_width_df.csv")

tables = {
    "corr_oas_dimension": corr_oas_dimension,
    "corr_skew_dimension": corr_skew_dimension,
    "corr_vix_dimension": corr_vix_dimension,
    "mfdfa_width_df": mfdfa_width_df,
}

for name, table in tables.items():
    print(f"\n===== {name} =====")
    print("shape:", table.shape)
    print("columns:", table.columns.tolist())
    display(table.head())

oas_corr = corr_oas_dimension.copy()
oas_corr["end_date"] = pd.to_datetime(oas_corr["end_date"])
oas_corr = oas_corr[["end_date", "corr_dim"]].rename(
    columns={"corr_dim": "oas_corr_dim"}
)

skew_corr = corr_skew_dimension.copy()
skew_corr["end_date"] = pd.to_datetime(skew_corr["end_date"])
skew_corr = skew_corr[["end_date", "corr_dim"]].rename(
    columns={"corr_dim": "skew_corr_dim"}
)

vix_corr = corr_vix_dimension.copy()
vix_corr["end_date"] = pd.to_datetime(vix_corr["end_date"])
vix_corr = vix_corr[["end_date", "corr_dim"]].rename(
    columns={"corr_dim": "vix_corr_dim"}
)

mfdfa_width = mfdfa_width_df.copy()
mfdfa_width["end_date"] = pd.to_datetime(mfdfa_width["end_date"])

mfdfa_width = mfdfa_width[
    [
        "end_date",
        "OAS_width",
        "VIX_width",
        "SKEW_width",
    ]
]

feature_table = (
    mfdfa_width
    .merge(oas_corr, on="end_date", how="inner")
    .merge(vix_corr, on="end_date", how="inner")
    .merge(skew_corr, on="end_date", how="inner")
    .sort_values("end_date")
    .reset_index(drop=True)
)

print(feature_table.head())

# Save feature table to CSV
feature_table.to_csv(
    data_path + "feature_table_oas_vix_skew.csv",
    index=False
)

## Logistic Regression Running ## -------------------------------------------------------------
# Build class that fit logistic regression model and evaluate the model performance
@dataclass
class LogisticRegressionPipeline:
    feature_table: pd.DataFrame
    market_table: pd.DataFrame
    event_table: pd.DataFrame

    feature_cols: list
    market_cols: list
    target_col: str = "y_next_event"

    feature_date_col: str = "end_date"
    market_date_col: str = "date"
    event_start_col: str = "start_date"

    lead_start: int = 1
    lead_end: int = 15
    tolerance_days: int = 3

    C: float = 1.0
    class_weight: str = "balanced"
    threshold: float = 0.5

    def _check_columns(self, table, cols, table_name):
        missing = [col for col in cols if col not in table.columns]
        if missing:
            raise KeyError(f"{table_name} missing columns: {missing}")

    def _make_event_label(self, dates):
        """
        for every feature date t：
        if there is event start_date within [t+lead_start, t+lead_end]，then y=1
        """
        dates = pd.to_datetime(dates)

        event_table = self.event_table.copy()
        event_table[self.event_start_col] = pd.to_datetime(event_table[self.event_start_col])
        event_starts = event_table[self.event_start_col].dropna()

        y = pd.Series(0, index=dates.index, dtype=int)

        for event_start in event_starts:
            valid_t_start = event_start - pd.Timedelta(days=self.lead_end)
            valid_t_end = event_start - pd.Timedelta(days=self.lead_start)
            mask = (dates >= valid_t_start) & (dates <= valid_t_end)
            y.loc[mask] = 1

        return y.rename(self.target_col)

    def build_input_table(self):
        feature_table = self.feature_table.copy()
        market_table = self.market_table.copy()
        feature_table[self.feature_date_col] = pd.to_datetime(feature_table[self.feature_date_col])
        market_table[self.market_date_col] = pd.to_datetime(market_table[self.market_date_col])

        self._check_columns(
            feature_table,
            [self.feature_date_col] + self.feature_cols,
            "feature_table")

        self._check_columns(
            market_table,
            [self.market_date_col] + self.market_cols,
            "market_table")

        market_for_model = (
            market_table[[self.market_date_col] + self.market_cols]
            .rename(columns={self.market_date_col: self.feature_date_col})
            .sort_values(self.feature_date_col))

        model_table = pd.merge_asof(
            feature_table.sort_values(self.feature_date_col),
            market_for_model,
            on=self.feature_date_col,
            direction="backward",
            tolerance=pd.Timedelta(f"{self.tolerance_days}D"))

        model_table = model_table.sort_values(self.feature_date_col).reset_index(drop=True)
        model_table[self.target_col] = self._make_event_label(model_table[self.feature_date_col])
        self.model_table = model_table
        return model_table

    def _evaluate(self, y_true, y_pred, y_prob, model_name):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        return {
            "model_name": model_name,
            "accuracy": accuracy_score(y_true, y_pred),

            "precision_1": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            "recall_1": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            "f1_1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),

            "precision_0": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
            "recall_0": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
            "f1_0": f1_score(y_true, y_pred, pos_label=0, zero_division=0),

            "auc": roc_auc_score(y_true, y_prob),
            "pr_auc": average_precision_score(y_true, y_prob),
            "brier": brier_score_loss(y_true, y_prob),

            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }

    def run_holdout(self, split_ratio=0.7, print_report=True):

        table = self.model_table.copy().sort_values(self.feature_date_col).reset_index(drop=True)

        split_idx = int(len(table) * split_ratio)

        y = table[self.target_col].astype(int)

        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        datasets = {
            "Market only": (
                table[self.market_cols].iloc[:split_idx],
                table[self.market_cols].iloc[split_idx:]),
            "Fractal/topological only": (
                table[self.feature_cols].iloc[:split_idx],
                table[self.feature_cols].iloc[split_idx:]),
            "Market + fractal/topological": (
                table[self.market_cols + self.feature_cols].iloc[:split_idx],
                table[self.market_cols + self.feature_cols].iloc[split_idx:])}

        results = []
        preds = {}

        for model_name, (X_train, X_test) in datasets.items():
            model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l2",
                C=self.C,
                class_weight=self.class_weight,
                max_iter=1000))
            model.fit(X_train, y_train)

            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= self.threshold).astype(int)

            results.append(self._evaluate(y_test, y_pred, y_prob, model_name))

            preds[model_name] = {
                "model": model,
                "y_prob": y_prob,
                "y_pred": y_pred}

            if print_report:
                print(f"\n=== {model_name} ===")
                print(confusion_matrix(y_test, y_pred))
                print(classification_report(y_test, y_pred, zero_division=0))

        return pd.DataFrame(results), preds

    def walk_forward(self,initial_train_size=1000,test_size=126,step_size=126,embargo=0):
        if not hasattr(self, "model_table"):
            self.build_input_table()

        table = self.model_table.copy()
        table = table.sort_values(self.feature_date_col).reset_index(drop=True)

        all_preds = []
        fold_results = []

        n = len(table)
        fold = 0
        train_end = initial_train_size

        while train_end + embargo + test_size <= n:
            test_start = train_end + embargo
            test_end = test_start + test_size

            train_data = table.iloc[:train_end].copy()
            test_data = table.iloc[test_start:test_end].copy()

            y_train = train_data[self.target_col].astype(int)
            y_test = test_data[self.target_col].astype(int)

            if y_train.nunique() < 2 or y_test.nunique() < 2:
                train_end += step_size
                continue

            datasets = {
                "Market only": (
                    train_data[self.market_cols],
                    test_data[self.market_cols],
                ),
                "Fractal/topological only": (
                    train_data[self.feature_cols],
                    test_data[self.feature_cols],
                ),
                "Market + fractal/topological": (
                    train_data[self.market_cols + self.feature_cols],
                    test_data[self.market_cols + self.feature_cols],
                )}

            fold_pred = {
                self.feature_date_col: test_data[self.feature_date_col].values,
                "y_true": y_test.values,
                "fold": fold}

            for model_name, (X_train, X_test) in datasets.items():
                model = make_pipeline(StandardScaler(),LogisticRegression(
                    penalty="l2",
                    C=self.C,
                    class_weight=self.class_weight,
                    max_iter=1000))
                model.fit(X_train, y_train)

                y_prob = model.predict_proba(X_test)[:, 1]
                y_pred = (y_prob >= self.threshold).astype(int)

                result = self._evaluate(y_test, y_pred, y_prob, model_name)
                result.update({
                    "fold": fold,
                    "train_start": train_data[self.feature_date_col].min(),
                    "train_end": train_data[self.feature_date_col].max(),
                    "test_start": test_data[self.feature_date_col].min(),
                    "test_end": test_data[self.feature_date_col].max(),
                    "train_n": len(train_data),
                    "test_n": len(test_data),
                    "test_positive_ratio": y_test.mean(),
                    "test_positive_n": int(y_test.sum()),
                    "test_negative_n": int((y_test == 0).sum())})

                fold_results.append(result)

                prefix = (
                    model_name
                    .lower()
                    .replace(" ", "_")
                    .replace("+", "plus")
                    .replace("/", "_"))

                fold_pred[f"{prefix}_pred"] = y_pred
                fold_pred[f"{prefix}_prob"] = y_prob

            all_preds.append(pd.DataFrame(fold_pred))

            fold += 1
            train_end += step_size

        fold_results = pd.DataFrame(fold_results)
        oos_pred_table = pd.concat(all_preds, ignore_index=True)

        self.fold_results = fold_results
        self.oos_pred_table = oos_pred_table

        return fold_results, oos_pred_table

    def walk_forward_kernel_ridge(
        self,
        initial_train_size=1000,
        test_size=126,
        step_size=126,
        embargo=0,
        alpha=1.0,
        gamma=0.1,
        kernel="rbf",
        use_sample_weight=True
    ):
        """
        Walk-forward evaluation using Kernel Ridge Regression.

        Notes:
        - KernelRidge does not output true probabilities.
        - y_score is the raw continuous score.
        - y_prob is clipped to [0, 1] only for thresholding and brier.
        - AUC / PR-AUC are computed using raw y_score.
        """
        if not hasattr(self, "model_table"):
            self.build_input_table()

        table = self.model_table.copy()
        table = table.sort_values(self.feature_date_col).reset_index(drop=True)

        all_preds = []
        fold_results = []

        n = len(table)
        fold = 0
        train_end = initial_train_size

        while train_end + embargo + test_size <= n:
            test_start = train_end + embargo
            test_end = test_start + test_size

            train_data = table.iloc[:train_end].copy()
            test_data = table.iloc[test_start:test_end].copy()

            y_train = train_data[self.target_col].astype(int)
            y_test = test_data[self.target_col].astype(int)

            if y_train.nunique() < 2 or y_test.nunique() < 2:
                train_end += step_size
                continue

            datasets = {
                "Market only": (
                    train_data[self.market_cols],
                    test_data[self.market_cols],
                ),
                "Fractal/topological only": (
                    train_data[self.feature_cols],
                    test_data[self.feature_cols],
                ),
                "Market + fractal/topological": (
                    train_data[self.market_cols + self.feature_cols],
                    test_data[self.market_cols + self.feature_cols],
                )
            }

            fold_pred = {
                self.feature_date_col: test_data[self.feature_date_col].values,
                "y_true": y_test.values,
                "fold": fold
            }

            for model_name, (X_train, X_test) in datasets.items():
                model = make_pipeline(
                    StandardScaler(),
                    KernelRidge(
                        kernel=kernel,
                        alpha=alpha,
                        gamma=gamma
                    )
                )

                if use_sample_weight and self.class_weight is not None:
                    sample_weight = compute_sample_weight(
                        class_weight=self.class_weight,
                        y=y_train
                    )
                    model.fit(
                        X_train,
                        y_train,
                        kernelridge__sample_weight=sample_weight
                    )
                else:
                    model.fit(X_train, y_train)

                y_score = model.predict(X_test)

                # KRR score is not probability; clip only for threshold + brier
                y_prob = np.clip(y_score, 0, 1)
                y_pred = (y_prob >= self.threshold).astype(int)

                result = self._evaluate(y_test, y_pred, y_prob, model_name)

                # Use raw continuous score for ranking metrics
                result["auc"] = roc_auc_score(y_test, y_score)
                result["pr_auc"] = average_precision_score(y_test, y_score)

                result.update({
                    "fold": fold,
                    "train_start": train_data[self.feature_date_col].min(),
                    "train_end": train_data[self.feature_date_col].max(),
                    "test_start": test_data[self.feature_date_col].min(),
                    "test_end": test_data[self.feature_date_col].max(),
                    "train_n": len(train_data),
                    "test_n": len(test_data),
                    "test_positive_ratio": y_test.mean(),
                    "test_positive_n": int(y_test.sum()),
                    "test_negative_n": int((y_test == 0).sum()),
                    "model_type": "KernelRidge",
                    "kernel": kernel,
                    "alpha": alpha,
                    "gamma": gamma,
                    "score_min": float(np.min(y_score)),
                    "score_max": float(np.max(y_score)),
                    "score_mean": float(np.mean(y_score))
                })

                fold_results.append(result)

                prefix = (
                    model_name
                    .lower()
                    .replace(" ", "_")
                    .replace("+", "plus")
                    .replace("/", "_")
                )

                fold_pred[f"{prefix}_pred"] = y_pred
                fold_pred[f"{prefix}_prob"] = y_prob
                fold_pred[f"{prefix}_score"] = y_score

            all_preds.append(pd.DataFrame(fold_pred))

            fold += 1
            train_end += step_size

        fold_results = pd.DataFrame(fold_results)
        oos_pred_table = pd.concat(all_preds, ignore_index=True)

        self.fold_results_krr = fold_results
        self.oos_pred_table_krr = oos_pred_table

        return fold_results, oos_pred_table




    def _make_width_rbf_logit(self,columns,rbf_col="OAS_width",gamma=1.0,n_components=20,random_state=0):
        columns = list(columns)

        linear_cols = [c for c in columns if c != rbf_col]
        transformers = []

        if linear_cols:
            transformers.append((
                "linear",
                StandardScaler(),
                linear_cols))

        if rbf_col in columns:
            transformers.append((
                "oas_width_rbf",
                make_pipeline(
                  StandardScaler(),
                  RBFSampler(
                    gamma=gamma,
                    n_components=n_components,
                    random_state=random_state)),[rbf_col]))

        preprocessor = ColumnTransformer(transformers=transformers,remainder="drop")

        model = Pipeline([
              ("features", preprocessor),
              ("logit", LogisticRegression(
                penalty="l2",
                C=self.C,
                class_weight=self.class_weight,
                max_iter=1000))])

        return model

    def walk_forward_width_rbf(
        self,
        initial_train_size=1000,
        test_size=126,
        step_size=126,
        embargo=0,
        rbf_col="OAS_width",
        gamma=1.0,
        n_components=20,
        random_state=0):
        if not hasattr(self, "model_table"):
          self.build_input_table()

        table = self.model_table.copy()
        table = table.sort_values(self.feature_date_col).reset_index(drop=True)

        all_preds = []
        fold_results = []

        n = len(table)
        fold = 0
        train_end = initial_train_size

        while train_end + embargo + test_size <= n:
          test_start = train_end + embargo
          test_end = test_start + test_size

          train_data = table.iloc[:train_end].copy()
          test_data = table.iloc[test_start:test_end].copy()

          y_train = train_data[self.target_col].astype(int)
          y_test = test_data[self.target_col].astype(int)

          if y_train.nunique() < 2 or y_test.nunique() < 2:
              train_end += step_size
              continue

          datasets = {
            "Market only": (
                train_data[self.market_cols],
                test_data[self.market_cols]),
            "Fractal/topological only": (
                train_data[self.feature_cols],
                test_data[self.feature_cols]),
            "Market + fractal/topological": (
                train_data[self.market_cols + self.feature_cols],
                test_data[self.market_cols + self.feature_cols])}

          fold_pred = {
            self.feature_date_col: test_data[self.feature_date_col].values,
            "y_true": y_test.values,
            "fold": fold}

          for model_name, (X_train, X_test) in datasets.items():

              model = self._make_width_rbf_logit(
                columns=X_train.columns,
                rbf_col=rbf_col,
                gamma=gamma,
                n_components=n_components,
                random_state=random_state)

              model.fit(X_train, y_train)

              y_prob = model.predict_proba(X_test)[:, 1]
              y_pred = (y_prob >= self.threshold).astype(int)
              result = self._evaluate(y_test, y_pred, y_prob, model_name)
              result.update({
                "fold": fold,
                "train_start": train_data[self.feature_date_col].min(),
                "train_end": train_data[self.feature_date_col].max(),
                "test_start": test_data[self.feature_date_col].min(),
                "test_end": test_data[self.feature_date_col].max(),
                "train_n": len(train_data),
                "test_n": len(test_data),
                "test_positive_ratio": y_test.mean(),
                "test_positive_n": int(y_test.sum()),
                "test_negative_n": int((y_test == 0).sum()),
                "model_type": "Logit + OAS_width_RBF",
                "rbf_col": rbf_col,
                "gamma": gamma,
                "n_components": n_components})

              fold_results.append(result)

              prefix = (
                model_name
                .lower()
                .replace(" ", "_")
                .replace("+", "plus")
                .replace("/", "_"))

              fold_pred[f"{prefix}_pred"] = y_pred
              fold_pred[f"{prefix}_prob"] = y_prob

          all_preds.append(pd.DataFrame(fold_pred))

          fold += 1
          train_end += step_size

        fold_results = pd.DataFrame(fold_results)
        oos_pred_table = pd.concat(all_preds, ignore_index=True)

        self.fold_results_rbf = fold_results
        self.oos_pred_table_rbf = oos_pred_table

        return fold_results, oos_pred_table

    def make_delta_panel(
        self,
        fold_results=None,
        baseline_model="Market only",
        full_model="Market + fractal/topological"):
        """
        Compare full_model against baseline_model fold by fold.

        Positive delta means full_model is better.
        For brier, lower is better, so delta_brier = baseline_brier - full_brier.
        """
        if fold_results is None:
            if not hasattr(self, "fold_results"):
                raise ValueError("No fold_results found. Run self.walk_forward() first.")
            fold_results = self.fold_results

        required_cols = {"fold", "model_name"}
        missing = required_cols - set(fold_results.columns)
        if missing:
            raise KeyError(f"fold_results missing columns: {missing}")

        model_names = set(fold_results["model_name"].unique())
        if baseline_model not in model_names:
            raise KeyError(f"baseline_model not found: {baseline_model}")
        if full_model not in model_names:
            raise KeyError(f"full_model not found: {full_model}")

        higher_better = [
            "accuracy",
            "auc",
            "pr_auc",
            "precision_1",
            "recall_1",
            "f1_1",
            "precision_0",
            "recall_0",
            "f1_0"
        ]

        lower_better = [
            "brier"
        ]

        meta_cols = [
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "train_n",
            "test_n",
            "test_positive_ratio",
            "test_positive_n",
            "test_negative_n"
        ]

        existing_meta_cols = [c for c in meta_cols if c in fold_results.columns]

        fold_index = sorted(fold_results["fold"].unique())
        delta = pd.DataFrame(index=pd.Index(fold_index, name="fold"))

        if existing_meta_cols:
            meta = (
                fold_results
                .sort_values("fold")
                .drop_duplicates("fold")
                .set_index("fold")[existing_meta_cols]
            )
            delta = delta.join(meta, how="left")

        wide = fold_results.set_index(["fold", "model_name"])

        for metric in higher_better:
            if metric not in fold_results.columns:
                continue

            metric_by_model = wide[metric].unstack("model_name")

            if baseline_model in metric_by_model.columns and full_model in metric_by_model.columns:
                delta[f"delta_{metric}"] = (
                    metric_by_model[full_model]
                    - metric_by_model[baseline_model]
                )

        for metric in lower_better:
            if metric not in fold_results.columns:
                continue

            metric_by_model = wide[metric].unstack("model_name")

            if baseline_model in metric_by_model.columns and full_model in metric_by_model.columns:
                delta[f"delta_{metric}"] = (
                    metric_by_model[baseline_model]
                    - metric_by_model[full_model]
                )

        delta_df = delta.reset_index()

        self.delta_df = delta_df
        self.delta_baseline_model = baseline_model
        self.delta_full_model = full_model

        return delta_df


    def summarize_delta(self, delta_df=None):
        """
        Summarize fold-level deltas.

        win_rate = share of folds where full model beats baseline.
        sign_test_p = one-sided sign test against 50% win rate.
        """
        if delta_df is None:
            if hasattr(self, "delta_df"):
                delta_df = self.delta_df
            else:
                delta_df = self.make_delta_panel()

        delta_cols = [c for c in delta_df.columns if c.startswith("delta_")]

        rows = []

        for col in delta_cols:
            x = delta_df[col].dropna()
            n = len(x)

            if n == 0:
                continue

            wins = int((x > 0).sum())

            rows.append({
                "metric": col.replace("delta_", ""),
                "mean_delta": x.mean(),
                "median_delta": x.median(),
                "win_rate": wins / n,
                "n_positive_folds": wins,
                "n_folds": n,
                "q25": x.quantile(0.25),
                "q75": x.quantile(0.75),
                "min": x.min(),
                "max": x.max(),
                "sign_test_p": binomtest(
                    wins,
                    n,
                    p=0.5,
                    alternative="greater"
                ).pvalue
            })

        delta_summary = (
            pd.DataFrame(rows)
            .sort_values(["win_rate", "median_delta"], ascending=False)
            .reset_index(drop=True)
        )

        self.delta_summary = delta_summary

        return delta_summary


    def plot_delta_over_time(self, metric, delta_df=None):
        """
        Plot fold-level improvement over time.

        Example:
            self.plot_delta_over_time("pr_auc")
            self.plot_delta_over_time("recall_1")
            self.plot_delta_over_time("recall_0")
            self.plot_delta_over_time("brier")
        """
        if delta_df is None:
            if hasattr(self, "delta_df"):
                delta_df = self.delta_df
            else:
                delta_df = self.make_delta_panel()

        col = f"delta_{metric}"

        if col not in delta_df.columns:
            raise KeyError(f"{col} not found in delta_df.")

        if "test_start" in delta_df.columns:
            x = delta_df["test_start"]
            xlabel = "test_start"
        else:
            x = delta_df["fold"]
            xlabel = "fold"

        plt.figure(figsize=(12, 4))
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.plot(x, delta_df[col], marker="o")
        plt.title(f"Fold-level improvement: {metric}")
        plt.xlabel(xlabel)
        plt.ylabel(f"Delta {metric}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_delta_precision_recall_by_class(self, class_label, delta_df=None):
        """
        Plot fold-level delta precision and delta recall for one class.

        class_label = 1: event / crisis class
        class_label = 0: calm / non-event class
        """
        if class_label not in [0, 1]:
            raise ValueError("class_label must be 0 or 1.")

        if delta_df is None:
            if hasattr(self, "delta_df"):
                delta_df = self.delta_df
            else:
                delta_df = self.make_delta_panel()

        precision_col = f"delta_precision_{class_label}"
        recall_col = f"delta_recall_{class_label}"

        missing = [c for c in [precision_col, recall_col] if c not in delta_df.columns]
        if missing:
            raise KeyError(f"Missing columns in delta_df: {missing}")

        if "test_start" in delta_df.columns:
            x = delta_df["test_start"]
            xlabel = "test_start"
        else:
            x = delta_df["fold"]
            xlabel = "fold"

        plt.figure(figsize=(12, 4))
        plt.axhline(0, linestyle="--", linewidth=1)

        plt.plot(
            x,
            delta_df[precision_col],
            marker="o",
            label=f"delta_precision_{class_label}"
        )

        plt.plot(
            x,
            delta_df[recall_col],
            marker="o",
            label=f"delta_recall_{class_label}"
        )

        plt.title(f"Fold-level improvement: class {class_label} precision vs recall")
        plt.xlabel(xlabel)
        plt.ylabel("Full model - Market only")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()



# Call datasets
#event_dir = "/content/drive/MyDrive/557project/event_tables"
event_dir = data_path.copy()

vix_level_events = pd.read_csv(f"{event_dir}vix_level_high_events.csv")
vix_delta_pos_events = pd.read_csv(f"{event_dir}vix_delta_pos_events.csv")
vix_delta_abs_events = pd.read_csv(f"{event_dir}vix_delta_abs_events.csv")
oas_level_events = pd.read_csv(f"{event_dir}oas_level_high_events.csv")
skew_level_events = pd.read_csv(f"{event_dir}skew_level_high_events.csv")
ofr_level_events = pd.read_csv(f"{event_dir}ofr_level_high_events.csv")
ofr_delta_abs_events = pd.read_csv(f"{event_dir}ofr_delta_abs_events.csv")
nfci_level_events = pd.read_csv(f"{event_dir}nfci_level_high_events.csv")
stlf_level_events = pd.read_csv(f"{event_dir}stlf_level_high_events.csv")

vix_table = pd.read_csv(f"{data_path}vix_table_clean.csv")
oas_table = pd.read_csv(f"{data_path}hy_oas.csv")
skew_table = pd.read_csv(f"{data_path}skew_table_clean.csv")
ofr_table = pd.read_csv(f"{data_path}labels.csv")
nfci_table = pd.read_csv(f"{data_path}NFCI.csv")
stlf_table = pd.read_csv(f"{data_path}STLFSI4.csv")

feature_table = pd.read_csv(f"{data_path}feature_table_oas_vix_skew.csv")

# Logistic regression - vix
vix_pipe = LogisticRegressionPipeline(feature_table=feature_table,
    market_table=vix_table,
    event_table=vix_level_events,
    feature_cols=["oas_corr_dim","OAS_width","SKEW_width"],
    market_cols=["Close"],
    market_date_col="date",
    event_start_col="start_date",lead_start=1,lead_end=10)

fold_results_vix, oos_pred_table_vi = vix_pipe.walk_forward(initial_train_size=1000,test_size=126,step_size=126,embargo=10)
delta_vix = vix_pipe.make_delta_panel()
print(delta_vix[["delta_recall_1","delta_precision_1","delta_recall_0","delta_precision_0","delta_brier","delta_pr_auc"]].describe())
# Save the results in txt file
with open(f"{output_path}vix_logreg_output.txt", "w") as f:
    f.write(delta_vix[["delta_recall_1","delta_precision_1","delta_recall_0","delta_precision_0","delta_brier","delta_pr_auc"]].describe().to_string())

delta_vix_feature = vix_pipe.make_delta_panel(baseline_model="Market only",full_model="Fractal/topological only")
print(delta_vix_feature[["delta_recall_1","delta_precision_1","delta_recall_0","delta_precision_0","delta_brier","delta_pr_auc"]].describe())
with open(f"{output_path}vix_logreg_output.txt", "a") as f:
    f.write(delta_vix_feature[["delta_recall_1","delta_precision_1","delta_recall_0","delta_precision_0","delta_brier","delta_pr_auc"]].describe().to_string())

# kernel ridge
# fold_results_krr_vix, oos_pred_table_krr_vix = vix_pipe.walk_forward_width_rbf(initial_train_size=1000,test_size=126,step_size=126,embargo=10,rbf_col="SKEW_width",gamma=0.1)
fold_results_krr_vix, oos_pred_table_krr_vix = vix_pipe.walk_forward_kernel_ridge(initial_train_size=1000,test_size=126,step_size=126,embargo=10,alpha=1.0,gamma=None)

# full_model="Fractal/topological only"
# "Market + fractal/topological"
delta_krr_vix = vix_pipe.make_delta_panel(fold_results=fold_results_krr_vix,baseline_model="Market only",full_model="Market + fractal/topological")
print(delta_krr_vix[["delta_recall_1","delta_precision_1","delta_recall_0","delta_precision_0","delta_brier","delta_pr_auc"]].describe())
with open(f"{output_path}vix_logreg_output.txt", "a") as f:
    f.write(delta_krr_vix[["delta_recall_1","delta_precision_1","delta_recall_0","delta_precision_0","delta_brier","delta_pr_auc"]].describe().to_string())

# Logistic regression - oas
lg_pipe = LogisticRegressionPipeline(feature_table=feature_table,
    market_table=oas_table,
    event_table=oas_level_events,
    feature_cols=["oas_corr_dim","OAS_width","SKEW_width"],
    market_cols=["BAMLH0A0HYM2"],
    market_date_col="observation_date",
    event_start_col="start_date",lead_start=1,lead_end=10)

model_table = lg_pipe.build_input_table()

print(model_table.shape)
# Save the result to txt file
with open(f"{output_path}oas_logreg_output.txt", "w") as f:
    f.write(f"Model table shape: {model_table.shape}\n")

holdout_results, holdout_preds = lg_pipe.run_holdout(split_ratio=0.7,print_report=True)
print(holdout_results)
with open(f"{output_path}oas_logreg_output.txt", "a") as f:
    f.write("Holdout results:\n")
    f.write(holdout_results.to_string(index=False))

fold_results, oos_pred_table = lg_pipe.walk_forward(initial_train_size=1000,test_size=126,step_size=126,embargo=10)
delta_df = lg_pipe.make_delta_panel()
lg_pipe.plot_delta_precision_recall_by_class(1)
lg_pipe.plot_delta_precision_recall_by_class(0)
print(delta_df[["delta_recall_1","delta_precision_1","delta_recall_0","delta_precision_0","delta_brier","delta_pr_auc"]].describe())
with open(f"{output_path}oas_logreg_output.txt", "a") as f:
    f.write("Walk-forward results:\n")
    f.write(delta_df[["delta_recall_1","delta_precision_1","delta_recall_0","delta_precision_0","delta_brier","delta_pr_auc"]].describe().to_string())

delta_feature_only = lg_pipe.make_delta_panel(baseline_model="Market only",full_model="Fractal/topological only")
lg_pipe.plot_delta_precision_recall_by_class(1,delta_df=delta_feature_only)
lg_pipe.plot_delta_precision_recall_by_class(0,delta_df=delta_feature_only)
print(delta_feature_only[["delta_recall_1","delta_precision_1","delta_recall_0","delta_precision_0","delta_brier","delta_pr_auc"]].describe())
with open(f"{output_path}oas_logreg_output.txt", "a") as f:
    f.write("Feature-only delta results:\n")
    f.write(delta_feature_only[["delta_recall_1","delta_precision_1","delta_recall_0","delta_precision_0","delta_brier","delta_pr_auc"]].describe().to_string())

# kernel ridge
# fold_results_krr, oos_pred_table_krr = lg_pipe.walk_forward_kernel_ridge(initial_train_size=1000,test_size=126,step_size=126,embargo=10,alpha=1.0,gamma=None)
fold_results_krr, oos_pred_table_krr = lg_pipe.walk_forward_width_rbf(initial_train_size=1000,test_size=126,step_size=126,embargo=10,rbf_col="SKEW_width",gamma=0.1)

# full_model="Fractal/topological only"
# "Market + fractal/topological"
delta_krr = lg_pipe.make_delta_panel(fold_results=fold_results_krr,baseline_model="Market only",full_model="Market + fractal/topological")
print(delta_krr[["delta_recall_1","delta_precision_1","delta_recall_0","delta_precision_0","delta_brier","delta_pr_auc"]].describe())
with open(f"{output_path}oas_logreg_output.txt", "a") as f:
    f.write("Kernel ridge delta results:\n")
    f.write(delta_krr[["delta_recall_1","delta_precision_1","delta_recall_0","delta_precision_0","delta_brier","delta_pr_auc"]].describe().to_string())

# logistic regression - skew
skew_pipe = LogisticRegressionPipeline(feature_table=feature_table,
    market_table=skew_table,
    event_table=skew_level_events,
    feature_cols=["oas_corr_dim","OAS_width","SKEW_width"],
    market_cols=["Close"],
    market_date_col="date",
    event_start_col="start_date",lead_start=1,lead_end=10)
model_table = skew_pipe.build_input_table()
fold_results, oos_pred_table = skew_pipe.walk_forward(initial_train_size=1000,test_size=126,step_size=126,embargo=10)
delta_skew = skew_pipe.make_delta_panel()
print(delta_skew[["delta_recall_1","delta_precision_1","delta_recall_0","delta_precision_0","delta_brier","delta_pr_auc"]].describe())
# Save the result to txt file
with open(f"{output_path}skew_logreg_output.txt", "w") as f:
    f.write(f"Model table shape: {model_table.shape}\n")
    f.write("Walk-forward results:\n")
    f.write(delta_skew[["delta_recall_1","delta_precision_1","delta_recall_0","delta_precision_0","delta_brier","delta_pr_auc"]].describe().to_string())

delta_feature_only_skew = skew_pipe.make_delta_panel(baseline_model="Market only",full_model="Fractal/topological only")
# skew_pipe.plot_delta_precision_recall_by_class(1,delta_df=delta_feature_only_skew)
# skew_pipe.plot_delta_precision_recall_by_class(0,delta_df=delta_feature_only_skew)
print(delta_feature_only_skew[["delta_recall_1","delta_precision_1","delta_recall_0","delta_precision_0","delta_brier","delta_pr_auc"]].describe())
with open(f"{output_path}skew_logreg_output.txt", "a") as f:
    f.write("Feature-only delta results:\n")
    f.write(delta_feature_only_skew[["delta_recall_1","delta_precision_1","delta_recall_0","delta_precision_0","delta_brier","delta_pr_auc"]].describe().to_string())

# kernel ridge
fold_results_krr_skew, oos_pred_table_krr_skew = skew_pipe.walk_forward_width_rbf(initial_train_size=1000,test_size=126,step_size=126,embargo=10,rbf_col="OAS_width",gamma=0.1)

# Market + fractal/topological
# full_model="Fractal/topological only"
delta_krr_skew = lg_pipe.make_delta_panel(fold_results=fold_results_krr_skew,baseline_model="Market only",full_model="Market + fractal/topological")
print(delta_krr_skew[["delta_recall_1","delta_precision_1","delta_recall_0","delta_precision_0","delta_brier","delta_pr_auc"]].describe())
with open(f"{output_path}skew_logreg_output.txt", "a") as f:
    f.write("Kernel ridge delta results:\n")
    f.write(delta_krr_skew[["delta_recall_1","delta_precision_1","delta_recall_0","delta_precision_0","delta_brier","delta_pr_auc"]].describe().to_string())

# logistic regression - ofr
ofr_pipe = LogisticRegressionPipeline(feature_table=feature_table,
    market_table=ofr_table,
    event_table=ofr_level_events,
    feature_cols=["oas_corr_dim","skew_corr_dim","SKEW_width"],
    market_cols=["OFR FSI"],
    market_date_col="Date",
    event_start_col="start_date",lead_start=1,lead_end=5)

fold_results_ofr, oos_pred_table_ofr = ofr_pipe.walk_forward(initial_train_size=1000,test_size=126,step_size=126,embargo=5)
delta_ofr = ofr_pipe.make_delta_panel()
print(delta_ofr[["delta_recall_1","delta_precision_1","delta_recall_0","delta_precision_0","delta_brier","delta_pr_auc"]].describe())
# Save the result to txt file
with open(f"{output_path}ofr_logreg_output.txt", "w") as f:
    f.write(f"Model table shape: {model_table.shape}\n")
    f.write("Walk-forward results:\n")
    f.write(delta_ofr[["delta_recall_1","delta_precision_1","delta_recall_0","delta_precision_0","delta_brier","delta_pr_auc"]].describe().to_string())

delta_feature_only_ofr = ofr_pipe.make_delta_panel(baseline_model="Market only",full_model="Fractal/topological only")
# skew_pipe.plot_delta_precision_recall_by_class(1,delta_df=delta_feature_only_skew)
# skew_pipe.plot_delta_precision_recall_by_class(0,delta_df=delta_feature_only_skew)
print(delta_feature_only_ofr[["delta_recall_1","delta_precision_1","delta_recall_0","delta_precision_0","delta_brier","delta_pr_auc"]].describe())
with open(f"{output_path}ofr_logreg_output.txt", "a") as f:
    f.write("Feature-only delta results:\n")
    f.write(delta_feature_only_ofr[["delta_recall_1","delta_precision_1","delta_recall_0","delta_precision_0","delta_brier","delta_pr_auc"]].describe().to_string())

# kernel
fold_results_krr_ofr, oos_pred_table_krr_ofr = ofr_pipe.walk_forward_width_rbf(initial_train_size=1000,test_size=126,step_size=126,embargo=5,rbf_col="SKEW_width",gamma=0.1)

# full_model="Fractal/topological only"
# "Market + fractal/topological"
delta_krr_ofr = ofr_pipe.make_delta_panel(fold_results=fold_results_krr_ofr,baseline_model="Market only",full_model="Market + fractal/topological")
print(delta_krr_ofr[["delta_recall_1","delta_precision_1","delta_recall_0","delta_precision_0","delta_brier","delta_pr_auc"]].describe())
with open(f"{output_path}ofr_logreg_output.txt", "a") as f:
    f.write("Kernel ridge delta results:\n")
    f.write(delta_krr_ofr[["delta_recall_1","delta_precision_1","delta_recall_0","delta_precision_0","delta_brier","delta_pr_auc"]].describe().to_string())