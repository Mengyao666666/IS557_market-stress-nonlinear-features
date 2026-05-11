'''
This script generates the Takens embedding features (correlation dimension) for the OAS, SKEW, and VIX time series, 
and then applies change point detection to identify potential regime shifts. 
The script also includes a placebo test to evaluate the significance of the detected change points in relation to known crisis events.
Used IPython file: takens_embedding.ipynb
AI Usage: The codes in this file are assisted by ChatGPT and GitHub Copilot.
'''

# Import necessary libraries
import numpy as np
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt

#from google.colab import drive

# File Paths
script_path = "final_code/"
data_path = "data/"
output_path = "outputs/"

#drive.mount('/content/drive')

# Load the dataset
df = pd.read_csv(data_path + "4columns_na_dropped.csv")

# Run the following codes if need to install
#!pip install nolds
#!pip install ruptures
#!pip install optuna

import ruptures as rpt
import optuna

vix_95 = pd.read_csv(data_path + "table2_VIX_rolling95_episodes.csv")
ofr_vix_95 = pd.read_csv(data_path + "table3_OFR_and_VIX_rolling95_episodes.csv")
ofr_90 = pd.read_csv(data_path + "table1_OFR_rolling90_episodes.csv")
vix_90 = pd.read_csv(data_path + "table2_VIX_rolling90_episodes.csv")
ofr_vix_90 = pd.read_csv(data_path + "table3_OFR_and_VIX_rolling90_episodes.csv")
vix_90_drop1 = vix_90[vix_90["n_days"] > 1].copy()
vix_90_drop1 = vix_90_drop1.reset_index(drop=True)

# Try univariate takens embedding (nolds)
import nolds

def rolling_corr_dim_plot(df,value_col,date_col="end_date",window_size=120,stride=5,
                     emb_dim=3,lag=5,fit="RANSAC",debug_plot=True):
    df = df.sort_values(date_col).copy()
    df[date_col] = pd.to_datetime(df[date_col])

    x = df[value_col].astype(float).to_numpy()
    dates = df[date_col].to_numpy()

    rows = []

    for start in range(0, len(x) - window_size + 1, stride):
        end = start + window_size
        w = x[start:end]

        if np.isnan(w).any():
            D = np.nan
        else:
            try:
                D = nolds.corr_dim(w,emb_dim=emb_dim,lag=lag,fit=fit,debug_plot=True)
            except Exception:
                D = np.nan

        rows.append({
            "window_start": dates[start],
            "end_date": dates[end - 1],
            "corr_dim": D,
            "window_size": window_size,
            "stride": stride,
            "emb_dim": emb_dim,
            "lag": lag
        })

    return pd.DataFrame(rows)

def rolling_corr_dim(df,value_col,date_col="end_date",window_size=120,stride=5,
                     emb_dim=3,lag=5,fit="RANSAC"):
    df = df.sort_values(date_col).copy()
    df[date_col] = pd.to_datetime(df[date_col])

    x = df[value_col].astype(float).to_numpy()
    dates = df[date_col].to_numpy()

    rows = []

    for start in range(0, len(x) - window_size + 1, stride):
        end = start + window_size
        w = x[start:end]

        if np.isnan(w).any():
            D = np.nan
        else:
            try:
                D = nolds.corr_dim(w,emb_dim=emb_dim,lag=lag,fit=fit)
            except Exception:
                D = np.nan

        rows.append({
            "window_start": dates[start],
            "end_date": dates[end - 1],
            "corr_dim": D,
            "window_size": window_size,
            "stride": stride,
            "emb_dim": emb_dim,
            "lag": lag
        })

    return pd.DataFrame(rows)

# Extract dimensions
dimension = rolling_corr_dim(df,value_col="OAS",date_col="Unnamed: 0",window_size=120,
                       stride=5,emb_dim=2,lag=3)

dimension_skew = rolling_corr_dim(df,value_col="SKEW",date_col="Unnamed: 0",window_size=120,
                       stride=5,emb_dim=2,lag=3)

dimension_skew_1 = rolling_corr_dim(df,value_col="SKEW",date_col="Unnamed: 0",window_size=120,
                       stride=1,emb_dim=2,lag=3)

dimension_1 = rolling_corr_dim(df,value_col="OAS",date_col="Unnamed: 0",window_size=120,
                       stride=1,emb_dim=2,lag=3)

dimension_vix_1 = rolling_corr_dim(df,value_col="VIX",date_col="Unnamed: 0",window_size=120,
                       stride=1,emb_dim=2,lag=3)

# Save as CSV
dimension_1.to_csv(data_path + "/dimension_oas_1.csv", index=False)
dimension_skew.to_csv(data_path + "/corr_dimension_skew.csv", index=False)
dimension_skew_1.to_csv(data_path + "/corr_dimension_skew_1.csv", index=False)
dimension_vix_1.to_csv(data_path + "/corr_dimension_vix_1.csv", index=False)
dimension.to_csv(data_path + "/corr_dimension.csv", index=False)


# Analysis futher -------------------------------------------------
from plotnine import *

# Based on dimension
dimension["end_date"] = pd.to_datetime(dimension["end_date"])
dimension["geom_risk"] = -dimension["corr_dim"]

(
    ggplot(dimension, aes(x="end_date", y="geom_risk")) +
    geom_line(alpha=0.8) +
    geom_point(size=0.7, alpha=0.5) +
    theme_minimal()
)

# Baed on dimension_1
dimension_1["end_date"] = pd.to_datetime(dimension_1["end_date"])

(
    ggplot(dimension_1, aes(x="end_date", y="corr_dim")) +
    geom_line(alpha=0.8) +
    geom_point(size=0.7, alpha=0.5) +
    theme_minimal()
)

# Test through CPD
def make_cp_df(mfdfa_df, feature_cols, date_col="end_date"):
    cp_df = mfdfa_df[[date_col] + feature_cols].copy()
    cp_df[date_col] = pd.to_datetime(cp_df[date_col])
    cp_df = cp_df.dropna().sort_values(date_col).reset_index(drop=True)
    return cp_df


def make_feature_matrix(cp_df, feature_cols):
    X = cp_df[feature_cols].values
    return X


def breakpoint_dates_from_bkps(cp_df, bkps, date_col="end_date"):
    dates = []

    for b in bkps[:-1]:
        if 1 <= b <= len(cp_df):
            dates.append(pd.Timestamp(cp_df[date_col].iloc[b - 1]))

    return pd.Series(
        sorted(pd.to_datetime(dates).unique()),
        name="breakpoint_date")

def run_pelt_cpd(mfdfa_df,feature_cols,date_col="end_date",model="rbf",pen=8,min_size=5,jump=1):
    cp_df = make_cp_df(mfdfa_df, feature_cols, date_col=date_col)
    X = make_feature_matrix(cp_df, feature_cols)

    algo = rpt.Pelt(model=model,min_size=min_size,jump=jump).fit(X)
    bkps = algo.predict(pen=pen)
    break_dates = breakpoint_dates_from_bkps(cp_df, bkps, date_col=date_col)

    return {
        "cp_df": cp_df,
        "X": X,
        "bkps": bkps,
        "break_dates": break_dates,
        "n_breaks": len(break_dates),
        "feature_cols": feature_cols,
        "model": model,
        "pen": pen}

def event_hit_rate(event_dates, break_dates, max_lead_days):
    event_dates = pd.to_datetime(pd.Series(event_dates)).dropna()
    event_dates = event_dates.sort_values().reset_index(drop=True)

    break_dates = pd.to_datetime(pd.Series(break_dates)).dropna()
    break_dates = break_dates.sort_values().reset_index(drop=True)

    hits = []

    for event_date in event_dates:
        pre_breaks = break_dates[
            (break_dates < event_date) &
            (break_dates >= event_date - pd.Timedelta(days=max_lead_days))]
        hits.append(1 if len(pre_breaks) > 0 else 0)

    return float(np.mean(hits)), hits

def random_event_placebo_test(cp_df,break_dates,real_event_dates,max_lead_days,
                              date_col="end_date",n_sim=1000,random_seed=42):
    rng = np.random.default_rng(random_seed)

    cp_dates = pd.to_datetime(cp_df[date_col]).sort_values().reset_index(drop=True)
    candidate_dates = pd.to_datetime(pd.Series(cp_dates.unique())).sort_values().reset_index(drop=True)

    break_dates = pd.to_datetime(pd.Series(break_dates)).dropna().sort_values().reset_index(drop=True)
    real_event_dates = pd.to_datetime(pd.Series(real_event_dates)).dropna().sort_values().reset_index(drop=True)

    real_hit_rate, real_hits = event_hit_rate(
        real_event_dates,
        break_dates,
        max_lead_days=max_lead_days)

    n_events = len(real_event_dates)

    if n_events > len(candidate_dates):
        raise ValueError("n_events is larger than number of candidate dates.")

    placebo_hit_rates = []

    for _ in range(n_sim):
        sampled_dates = rng.choice(candidate_dates.values, size=n_events, replace=False)
        sampled_dates = pd.to_datetime(pd.Series(sampled_dates)).sort_values().reset_index(drop=True)

        hit_rate, _ = event_hit_rate(
            sampled_dates,
            break_dates,
            max_lead_days=max_lead_days)

        placebo_hit_rates.append(hit_rate)

    placebo_hit_rates = np.array(placebo_hit_rates)

    return {
        "test_type": "random_event",
        "window_days": max_lead_days,
        "real_hit_rate": real_hit_rate,
        "real_hits": real_hits,
        "placebo_hit_rates": placebo_hit_rates,
        "p_value": float(np.mean(placebo_hit_rates >= real_hit_rate)),
        "placebo_mean": float(np.mean(placebo_hit_rates)),
        "placebo_std": float(np.std(placebo_hit_rates)),
        "placebo_q95": float(np.quantile(placebo_hit_rates, 0.95))}

def random_break_placebo_test(cp_df,break_dates,real_event_dates,max_lead_days,
                              date_col="end_date",n_sim=1000,random_seed=42):
    rng = np.random.default_rng(random_seed)

    cp_dates = pd.to_datetime(cp_df[date_col]).sort_values().reset_index(drop=True)
    candidate_dates = pd.to_datetime(pd.Series(cp_dates.unique())).sort_values().reset_index(drop=True)

    break_dates = pd.to_datetime(pd.Series(break_dates)).dropna().sort_values().reset_index(drop=True)
    real_event_dates = pd.to_datetime(pd.Series(real_event_dates)).dropna().sort_values().reset_index(drop=True)

    real_hit_rate, real_hits = event_hit_rate(
        real_event_dates,
        break_dates,
        max_lead_days=max_lead_days)

    n_breaks = len(break_dates)

    if n_breaks > len(candidate_dates):
        raise ValueError("n_breaks is larger than number of candidate dates.")

    placebo_hit_rates = []

    for _ in range(n_sim):
        fake_break_dates = rng.choice(candidate_dates.values, size=n_breaks, replace=False)
        fake_break_dates = pd.to_datetime(pd.Series(fake_break_dates)).sort_values().reset_index(drop=True)

        hit_rate, _ = event_hit_rate(
            real_event_dates,
            fake_break_dates,
            max_lead_days=max_lead_days)

        placebo_hit_rates.append(hit_rate)

    placebo_hit_rates = np.array(placebo_hit_rates)

    return {
        "test_type": "random_break",
        "window_days": max_lead_days,
        "real_hit_rate": real_hit_rate,
        "real_hits": real_hits,
        "placebo_hit_rates": placebo_hit_rates,
        "p_value": float(np.mean(placebo_hit_rates >= real_hit_rate)),
        "placebo_mean": float(np.mean(placebo_hit_rates)),
        "placebo_std": float(np.std(placebo_hit_rates)),
        "placebo_q95": float(np.quantile(placebo_hit_rates, 0.95))}


def placebo_summary_table(cp_df,break_dates,real_event_dates,windows=(30, 60),
                          date_col="end_date",n_sim=1000,random_seed=42,tests=("random_event", "random_break")):
    rows = []

    for days in windows:
        if "random_event" in tests:
            res = random_event_placebo_test(
                cp_df=cp_df,
                break_dates=break_dates,
                real_event_dates=real_event_dates,
                max_lead_days=days,
                date_col=date_col,
                n_sim=n_sim,
                random_seed=random_seed)

            rows.append({
                "test_type": "random_event",
                "window_days": days,
                "real_hit_rate": res["real_hit_rate"],
                "placebo_mean": res["placebo_mean"],
                "placebo_std": res["placebo_std"],
                "placebo_q95": res["placebo_q95"],
                "p_value": res["p_value"],})

        if "random_break" in tests:
            res = random_break_placebo_test(
                cp_df=cp_df,
                break_dates=break_dates,
                real_event_dates=real_event_dates,
                max_lead_days=days,
                date_col=date_col,
                n_sim=n_sim,
                random_seed=random_seed)

            rows.append({
                "test_type": "random_break",
                "window_days": days,
                "real_hit_rate": res["real_hit_rate"],
                "placebo_mean": res["placebo_mean"],
                "placebo_std": res["placebo_std"],
                "placebo_q95": res["placebo_q95"],
                "p_value": res["p_value"],})

    return pd.DataFrame(rows)


def run_cpd_placebo_pipeline(mfdfa_df,feature_cols,real_event_dates,date_col="end_date",
                             model="rbf",pen=8,min_size=5,jump=1,windows=(30, 60),
                             n_sim=1000,random_seed=42,tests=("random_event", "random_break")):

    cpd_res = run_pelt_cpd(mfdfa_df=mfdfa_df,feature_cols=feature_cols,
                           date_col=date_col,model=model,pen=pen,min_size=min_size,jump=jump)

    summary = placebo_summary_table(cp_df=cpd_res["cp_df"],break_dates=cpd_res["break_dates"],
                                    real_event_dates=real_event_dates,windows=windows,date_col=date_col,n_sim=n_sim,random_seed=random_seed,tests=tests)

    return {
        **cpd_res,
        "summary": summary}

def run_ablation_cpd_pipeline(mfdfa_df,feature_configs,real_event_dates,date_col="end_date",
                              model="rbf",pen=8,min_size=5,jump=1,
                              windows=(30, 60),n_sim=1000,random_seed=42,tests=("random_event", "random_break")):

    all_results = {}
    all_summaries = []

    for config_name, cols in feature_configs.items():
        # print(f"Running config: {config_name}, n_features={len(cols)}")

        res = run_cpd_placebo_pipeline(mfdfa_df=mfdfa_df,feature_cols=cols,
                                       real_event_dates=real_event_dates,date_col=date_col,
                                       model=model,pen=pen,min_size=min_size,jump=jump,windows=windows,n_sim=n_sim,random_seed=random_seed,tests=tests)

        summary = res["summary"].copy()
        summary.insert(0, "config", config_name)
        summary.insert(1, "n_features", len(cols))
        summary.insert(2, "n_breaks", res["n_breaks"])

        all_summaries.append(summary)
        all_results[config_name] = res

    ablation_summary = pd.concat(all_summaries, ignore_index=True)

    return all_results, ablation_summary


# Ablation test 1
feature_configs = {
    "corr_dim": ["corr_dim"]
}

ablation_results1, ablation_summary1 = run_ablation_cpd_pipeline(
    mfdfa_df=dimension,
    feature_configs=feature_configs,
    real_event_dates=vix_95["onset"],
    date_col="end_date",
    model="l2",
    pen=0.2,
    windows=(15, 30),
    n_sim=1000,
    random_seed=42
)

print(ablation_summary1)
# Save ablation summary to a text file
with open(output_path + "takens_embedding_summaries.txt", "w") as f:    
    f.write(ablation_summary1.to_string(index=False))

# Ablation test 2
feature_configs = {
    "corr_dim": ["corr_dim"]
}

ablation_results2, ablation_summary2 = run_ablation_cpd_pipeline(
    mfdfa_df=dimension,
    feature_configs=feature_configs,
    real_event_dates=ofr_90["onset"],
    date_col="end_date",
    model="l2",
    pen=0.05,
    windows=(15, 30, 45),
    n_sim=1000,
    random_seed=42
)

print(ablation_summary2)
with open(output_path + "takens_embedding_summaries.txt", "a") as f:    
    f.write("\n\n")
    f.write(ablation_summary2.to_string(index=False))

# Ablation test 3
ablation_results3, ablation_summary3 = run_ablation_cpd_pipeline(
    mfdfa_df=dimension,
    feature_configs=feature_configs,
    real_event_dates=ofr_vix_95["onset"],
    date_col="end_date",
    model="l2",
    pen=0.05,
    windows=(15, 30, 45),
    n_sim=1000,
    random_seed=42
)

print(ablation_summary3)
with open(output_path + "takens_embedding_summaries.txt", "a") as f:    
    f.write("\n\n")
    f.write(ablation_summary3.to_string(index=False))

# Ablation test 4
ablation_results4, ablation_summary4 = run_ablation_cpd_pipeline(
    mfdfa_df=dimension,
    feature_configs=feature_configs,
    real_event_dates=ofr_vix_90["onset"],
    date_col="end_date",
    model="l2",
    pen=0.1,
    windows=(15, 30, 45, 60),
    n_sim=1000,
    random_seed=42
)

print(ablation_summary4)
with open(output_path + "takens_embedding_summaries.txt", "a") as f:    
    f.write("\n\n")
    f.write(ablation_summary4.to_string(index=False))

# Change Point Detection result
rows = []

for pen in [0.05, 0.1, 0.2]:
    bkps = rpt.Pelt(model="l2", min_size=5).fit(X).predict(pen=pen)

    for b in bkps[:-1]:
        idx = b - 1
        rows.append({
            "pen": pen,
            "break_idx": idx,
            "break_date": dimension.iloc[idx]["end_date"],
            "corr_dim": dimension.iloc[idx]["corr_dim"]
        })

break_df = pd.DataFrame(rows)
print(break_df)
with open(output_path + "takens_embedding_breakpoints.csv", "w") as f:
    f.write("pen,break_idx,break_date,corr_dim\n")    
    f.write(break_df.to_string(index=False))

break_by_pen = (
    break_df
    .assign(break_date=pd.to_datetime(break_df["break_date"]).dt.strftime("%Y-%m-%d"))
    .groupby("pen")["break_date"]
    .apply(lambda s: ", ".join(s))
    .reset_index(name="break_dates")
)

print(break_by_pen)
with open(output_path + "takens_embedding_breakpoints.csv", "a") as f:
    f.write("\n\nBreak dates by pen:\n")    
    f.write(break_by_pen.to_string(index=False))
    f.write("\n\n")

#
dim_cpd = dimension.dropna(subset=["corr_dim"]).sort_values("end_date").copy()

X = dim_cpd[["corr_dim"]].values

for pen in [0.05, 0.1, 0.2, 0.5, 1, 2, 3, 5, 8]:
    bkps = rpt.Pelt(model="rbf", min_size=5).fit(X).predict(pen=pen)
    print("pen =", pen, "n_breaks =", len(bkps) - 1)
    with open(output_path + "takens_embedding_breakpoints.csv", "a") as f:    
        f.write(f"\npen = {pen}, n_breaks = {len(bkps) - 1}")