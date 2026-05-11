'''
The script implements a change point detection (CPD) pipeline using the PELT algorithm from the ruptures library. 
It applies CPD to the MFDFA features extracted from financial time series data, 
and evaluates the significance of detected change points in relation to known crisis events through placebo tests. 
The script also includes an ablation study to assess the contribution of different feature sets to the CPD performance.
Codes from: Ablation_test.ipynb
AI Usage: The codes in this file are assisted by ChatGPT and GitHub Copilot.
'''
#!pip install ruptures
#!pip install optuna
import pandas as pd
import numpy as np
import ruptures as rpt
import optuna

#from google.colab import drive
#drive.mount('/content/drive')

# File Paths
script_path = "/content/drive/MyDrive/557project/final_code/"
data_path = "/content/drive/MyDrive/557project/data/"
output_path = "/content/drive/MyDrive/557project/outputs/"

# List of features for each group
G_SKEW_width = ["SKEW_width"]
G_SKEW = ["SKEW_width", "SKEW_hq_mean"]
G_OAS = ["OAS_width", "OAS_hq_mean"]
G_SPREAD = ["Spread_width", "Spread_hq_mean"]

feature_configs = {
    "SKEW_width": G_SKEW_width,
    "SKEW_only": G_SKEW,
    "OAS_only": G_OAS,
    "Spread_only": G_SPREAD,

    "SKEW_OAS": G_SKEW + G_OAS,
    "SKEW_Spread": G_SKEW + G_SPREAD,
    "OAS_Spread": G_OAS + G_SPREAD,

    "All": G_SKEW + G_OAS + G_SPREAD,}

# Load the dataset
mfdfa_df = pd.read_csv(f"{data_path}mfdfa_width_df.csv")

ofr_90 = pd.read_csv(f"{data_path}table1_OFR_rolling90_episodes.csv")
vix_90 = pd.read_csv(f"{data_path}table2_VIX_rolling90_episodes.csv")
ofr_vix_90 = pd.read_csv(f"{data_path}table3_OFR_and_VIX_rolling90_episodes.csv")

vix_90_drop1 = vix_90[vix_90["n_days"] > 1].copy()
vix_90_drop1 = vix_90_drop1.reset_index(drop=True)

vix_95 = pd.read_csv(f"{data_path}table2_VIX_rolling95_episodes.csv")
ofr_vix_95 = pd.read_csv(f"{data_path}table3_OFR_and_VIX_rolling95_episodes.csv")

# CPD Pipeline
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
        print(f"Running config: {config_name}, n_features={len(cols)}")

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

## Ablation test 1
ablation_results1, ablation_summary1 = run_ablation_cpd_pipeline(
    mfdfa_df=mfdfa_df,
    feature_configs=feature_configs,
    real_event_dates=ofr_90["onset"],
    model="rbf",
    pen=8,
    windows=(75, 90),
    n_sim=1000,
    random_seed=42
)

print(ablation_summary1)
# Save the ablation summary to txt file, not csv file in output folder
with open(f"{output_path}ablation_summaries.txt", "w") as f:
    f.write(ablation_summary1.to_string(index=False))

## Ablation test 2
ablation_results2, ablation_summary2 = run_ablation_cpd_pipeline(
    mfdfa_df=mfdfa_df,
    feature_configs=feature_configs,
    real_event_dates=vix_90["onset"],
    model="rbf",
    pen=8,
    windows=(30, 60),
    n_sim=1000,
    random_seed=42
)

print(ablation_summary2)
with open(f"{output_path}ablation_summaries.txt", "a") as f:    
    f.write("\n\n")
    f.write(ablation_summary2.to_string(index=False))

## Ablation test 3
ablation_results3, ablation_summary3 = run_ablation_cpd_pipeline(
    mfdfa_df=mfdfa_df,
    feature_configs=feature_configs,
    real_event_dates=ofr_vix_90["onset"],
    model="rbf",
    pen=8,
    windows=(75, 90),
    n_sim=1000,
    random_seed=42
)

print(ablation_summary3)
with open(f"{output_path}ablation_summaries.txt", "a") as f:    
    f.write("\n\n")
    f.write(ablation_summary3.to_string(index=False))

# Other feature settings
feature_configs_2 = {
    "SKEW_only": G_SKEW,
    "OAS_only": G_OAS,
    "Spread_only": G_SPREAD,
    "SKEW_OAS": G_SKEW + G_OAS,
    "SKEW_Spread": G_SKEW + G_SPREAD,
    "OAS_Spread": G_OAS + G_SPREAD}

## Ablation test 4
ablation_results4, ablation_summary4 = run_ablation_cpd_pipeline(
    mfdfa_df=mfdfa_df,
    feature_configs=feature_configs_2,
    real_event_dates=vix_90_drop1["onset"],
    model="rbf",
    pen=8,
    windows=(30, 60),
    n_sim=1000,
    random_seed=42
)

print(ablation_summary4)
with open(f"{output_path}ablation_summaries.txt", "a") as f:    
    f.write("\n\n")
    f.write(ablation_summary4.to_string(index=False))
    
## Ablation test 5
ablation_results5, ablation_summary5 = run_ablation_cpd_pipeline(
    mfdfa_df=mfdfa_df,
    feature_configs=feature_configs_2,
    real_event_dates=vix_95["onset"],
    model="rbf",
    pen=8,
    windows=(30, 60),
    n_sim=1000,
    random_seed=42
)

print(ablation_summary5)
with open(f"{output_path}ablation_summaries.txt", "a") as f:    
    f.write("\n\n")
    f.write(ablation_summary5.to_string(index=False))

## Ablation test 6
ablation_results6, ablation_summary6 = run_ablation_cpd_pipeline(
    mfdfa_df=mfdfa_df,
    feature_configs=feature_configs_2,
    real_event_dates=ofr_vix_95["onset"],
    model="rbf",
    pen=8,
    windows=(30, 60),
    n_sim=1000,
    random_seed=42
)

print(ablation_summary6)
with open(f"{output_path}ablation_summaries.txt", "a") as f:    
    f.write("\n\n")
    f.write(ablation_summary6.to_string(index=False))

# Raw data features
raw_df = pd.read_csv(f"{data_path}4columns_na_dropped.csv")
raw_df = raw_df.rename(columns={raw_df.columns[0]: "Date"})
raw_df["Date"] = pd.to_datetime(raw_df["Date"])

feature_raw = {
    "SKEW_raw": ["SKEW"],
    "OAS_raw": ["OAS"],
    "Spread_raw": ["Spread"],

    "SKEW_OAS_raw": ["SKEW", "OAS"],
    "SKEW_Spread_raw": ["SKEW", "Spread"],
    "OAS_Spread_raw": ["OAS", "Spread"]}

## Ablation test 7
ablation_results7, ablation_summary7 = run_ablation_cpd_pipeline(
    mfdfa_df=raw_df,
    feature_configs=feature_raw,
    real_event_dates=ofr_vix_95["onset"],
    date_col="Date",
    model="rbf",
    pen=8,
    windows=(30, 60),
    n_sim=1000,
    random_seed=42
)

print(ablation_summary7)
with open(f"{output_path}ablation_summaries.txt", "a") as f:    
    f.write("\n\n")
    f.write(ablation_summary7.to_string(index=False))

## Ablation test 8
ablation_results8, ablation_summary8 = run_ablation_cpd_pipeline(
    mfdfa_df=raw_df,
    feature_configs=feature_raw,
    real_event_dates=vix_95["onset"],
    date_col="Date",
    model="rbf",
    pen=8,
    windows=(30, 60),
    n_sim=1000,
    random_seed=42
)

print(ablation_summary8)
with open(f"{output_path}ablation_summaries.txt", "a") as f:    
    f.write("\n\n")
    f.write(ablation_summary8.to_string(index=False))

## Ablation test 9
feature_one = {
    "SKEW_width": ["SKEW_width"]}

ablation_results9, ablation_summary9 = run_ablation_cpd_pipeline(
    mfdfa_df=mfdfa_df,
    feature_configs=feature_one,
    real_event_dates=ofr_90["onset"],
    model="rbf",
    pen=8,
    windows=(60, 75, 90),
    n_sim=1000,
    random_seed=42
)

print(ablation_summary9)
with open(f"{output_path}ablation_summaries.txt", "a") as f:    
    f.write("\n\n")
    f.write(ablation_summary9.to_string(index=False))

## Ablation test 10
feature_only = {
    "SKEW_raw": ["SKEW"]}

ablation_results10, ablation_summary10 = run_ablation_cpd_pipeline(
    mfdfa_df=raw_df,
    feature_configs=feature_only,
    real_event_dates=ofr_90["onset"],
    date_col="Date",
    model="rbf",
    pen=8,
    windows=(60, 75, 90),
    n_sim=1000,
    random_seed=42
)

print(ablation_summary10)
with open(f"{output_path}ablation_summaries.txt", "a") as f:    
    f.write("\n\n")
    f.write(ablation_summary10.to_string(index=False))

## Ablation test 11
ablation_results11, ablation_summary11 = run_ablation_cpd_pipeline(
    mfdfa_df=raw_df,
    feature_configs=feature_only,
    real_event_dates=ofr_90["onset"],
    date_col="Date",
    model="l2",
    pen=8,
    windows=(60, 75, 90),
    n_sim=1000,
    random_seed=42
)

print(ablation_summary11)
with open(f"{output_path}ablation_summaries.txt", "a") as f:    
    f.write("\n\n")
    f.write(ablation_summary11.to_string(index=False))