'''
The script generates the MFDFA features (spectrum width, mean, and asymmetry) for the VIX, SKEW, OAS, and Spread time series using a rolling window approach.
Codes from: spectrum width.ipynb
AI Usage: The codes in this file are assisted by ChatGPT and GitHub Copilot.
'''
# !pip install MFDFA

# Import necessary libraries
import pandas as pd
import numpy as np
from MFDFA import MFDFA
from datetime import datetime as dt
#import matplotlib.pyplot as plt

#from google.colab import drive
#drive.mount('/content/drive')

# File Paths
script_path = "final_code/"
data_path = "data/"
output_path = "outputs/"

# Load the datasets
df = pd.read_csv(data_path + "4columns_na_dropped.csv")
df["Date"] = pd.to_datetime(df["Unnamed: 0"])
df = df.set_index("Date")

#### About MFDFA
def run_mfdfa_rolling(df: pd.DataFrame,column: str,window: int = 180,step: int = 1,use_diff: bool = False,q_list: np.ndarray = None,order: int = 2,
    min_lag: int = 4,
    max_lag: int = None,
    n_lags: int = 20,
    min_valid_points: int = 40,
):

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index should be included in DatetimeIndex")

    if column not in df.columns:
        raise ValueError(f" {column} not in df")

    if q_list is None:
        q_list = np.array([-4, -3, -2, -1, 1, 2, 3, 4], dtype=float)

    series = df[column].copy()

    if use_diff:
        series = series.diff()

    def _compute_hq(series_values: np.ndarray):
        x = np.asarray(series_values, dtype=float)
        x = x[np.isfinite(x)]

        if len(x) < min_valid_points:
            return None

        local_max_lag = max_lag
        if local_max_lag is None:
            local_max_lag = min(len(x) // 4, 50)

        if local_max_lag <= min_lag:
            return None

        lag = np.unique(
            np.logspace(np.log10(min_lag), np.log10(local_max_lag), n_lags).astype(int)
        )
        lag = lag[lag > 1]

        if len(lag) < 2:
            return None

        lag, dfa = MFDFA(x, lag=lag, q=q_list, order=order)

        hq = np.empty(len(q_list))
        for i in range(len(q_list)):
            y = dfa[:, i]
            valid = np.isfinite(y) & (y > 0)
            if valid.sum() < 2:
                return None
            coeffs = np.polyfit(np.log(lag[valid]), np.log(y[valid]), 1)
            hq[i] = coeffs[0]

        return {
            "lag": lag,
            "dfa": dfa,
            "hq": hq,
        }

    rows = []

    for end in range(window - 1, len(series), step):
        # right-aligned rolling window
        window_series = series.iloc[end - window + 1 : end + 1]
        result = _compute_hq(window_series.values)

        if result is None:
            rows.append({
                "end_idx": end,
                "end_date": series.index[end],
                f"{column}_width": np.nan,
                f"{column}_hq_mean": np.nan,
                f"{column}_hq_asym": np.nan,
                f"{column}_hq": None,
            })
            continue

        hq = result["hq"]

        width = float(np.max(hq) - np.min(hq))
        hq_mean = float(np.mean(hq))
        hq_asym = float(np.mean(hq[q_list < 0]) - np.mean(hq[q_list > 0]))

        rows.append({
            "end_idx": end,
            "end_date": series.index[end],
            f"{column}_width": width,
            f"{column}_hq_mean": hq_mean,
            f"{column}_hq_asym": hq_asym,
            f"{column}_hq": hq,
        })

    return pd.DataFrame(rows)

# Get MFDFA features for each column

cols = ["VIX", "SKEW", "OAS", "Spread"]

mfdfa_dfs = []
for col in cols:
    tmp = run_mfdfa_rolling(
        df=df,
        column=col,
        window=180,
        step=1,
        use_diff=False
    )
    mfdfa_dfs.append(tmp.drop(columns=["end_idx"]))

# Save the results as CSV
# window = 180
from functools import reduce

width_df = reduce(
    lambda left, right: pd.merge(left, right, on="end_date", how="outer"),
    mfdfa_dfs
)

width_df.to_csv(data_path + "mfdfa_width_df.csv", index=False)