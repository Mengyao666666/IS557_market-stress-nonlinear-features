'''
This code is responsible for acquiring the necessary data for our analysis. 
We will be using the yfinance library and url from FRED and financialresearch.gov to fetch financial data and pandas to handle data manipulation. 
Since FRED Data does not allow reproduce or store data, we leave this script to be run by users to acquire the data and save it in the data folder.

AI Usage: This code was generated with the assistance of AI, which helped in structuring the data acquisition process and ensuring that the correct libraries and methods were used to fetch and store the data efficiently.
'''
import pandas as pd
import yfinance as yf

data_path = "../data/"
output_path = "../outputs/"

# Acquire Spread 10 year to 2 year data from FRED, in CSV format that already provided
df_spread = pd.read_csv('https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23ebf3fb&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1320&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=T10Y2Y&scale=left&cosd=2000-01-01&coed=2026-03-09&line_color=%230073e6&link_values=false&line_style=solid&mark_type=none&mw=3&lw=3&ost=-99999&oet=99999&mma=0&fml=a&fq=Daily&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2026-05-11&revision_date=2026-05-11&nd=1976-06-01')
# Save the data to a CSV file
df_spread.to_csv(data_path + 'spread_10y2y.csv', index=False)

# This is very unfortunate, but FRED does not provide High Yeild OAS data older than 3 years.
# Please have contact with FRED to acquire the data, and save it in the data folder as hy_oas.csv
df_oas = pd.read_csv(data_path + 'hy_oas.csv')

# Acquire SKEW data from CBOE, from yfinance library, and save it in the data folder as CBOE_SKEW_Full_2000_2026.csv
df_skew = yf.download("^SKEW", start="2000-01-01", end="2026-03-09")
df_skew.to_csv(data_path + 'CBOE_SKEW_Full_2000_2026.csv')

# Clean SKEW data by set the first column as date, and save it in the data folder as skew_table_clean.csv
df_skew_clean = df_skew.reset_index().iloc[:].copy()
df_skew_clean.columns = ["date", "Close", "High", "Low", "Open", "Volume"]
df_skew_clean.to_csv(data_path + 'skew_table_clean.csv', index=False)

# Acquire VIX data from CBOE, from yfinance library, and save it in the data folder as VIX.csv
df_vix = yf.download("^VIX", start="2000-01-01", end="2026-03-09")
df_vix.to_csv(data_path + 'VIX.csv')

# Clean VIX data by set the first column as date, and save it in the data folder as vix_table_clean.csv
df_vix_clean = df_vix.reset_index().iloc[:].copy()
df_vix_clean.columns = ["date", "Close", "High", "Low", "Open", "Volume"]
df_vix_clean.to_csv(data_path + 'vix_table_clean.csv', index=False)

# Data Cleaning to get 4columns_na_dropped.csv
dataframes = {
    "SKEW": df_skew,
    "VIX": df_vix,
    "10y-2y Spread": df_spread,
    "HY OAS": df_oas
}

# Extract close as Series (yfinance can return MultiIndex columns)
skew_close = (
    df_skew.xs("Close", axis=1, level=0).iloc[:, 0].rename("SKEW")
    if isinstance(df_skew.columns, pd.MultiIndex)
    else df_skew["Close"].rename("SKEW")
)

vix_close = (
    df_vix.xs("Close", axis=1, level=0).iloc[:, 0].rename("VIX")
    if isinstance(df_vix.columns, pd.MultiIndex)
    else df_vix["Close"].rename("VIX")
)

# Convert FRED/OAS dates to datetime index
spread_series = (
    df_spread.assign(observation_date=pd.to_datetime(df_spread["observation_date"]))
    .set_index("observation_date")["T10Y2Y"]
    .rename("Spread")
)

oas_series = (
    df_oas.assign(observation_date=pd.to_datetime(df_oas["observation_date"]))
    .set_index("observation_date")["BAMLH0A0HYM2"]
    .rename("OAS")
)

# Combine into one table
data_final = pd.concat([skew_close, vix_close, spread_series, oas_series], axis=1).sort_index()

data_model = data_final.dropna().copy()
data_model.to_csv(data_path + '4columns_na_dropped.csv')

# Download label from financialresearch.gov, and save it in the data folder as fsi.csv
label_df = pd.read_csv('https://www.financialresearch.gov/financial-stress-index/data/fsi.csv?')
label_df['Date'] = pd.to_datetime(label_df['Date'])
label_df[label_df['Date'] < pd.to_datetime('2026-03-06')].to_csv(data_path + 'labels.csv')

# SPX
spx = yf.download("^GSPC", start="2000-01-01", end="2026-03-06")
if isinstance(spx.columns, pd.MultiIndex):
    spx.columns = spx.columns.get_level_values(0)
spx = spx.reset_index()
spx.columns = [c.lower() for c in spx.columns]
spx_clean = spx[["date", "close"]].rename(columns={"close": "spx"})
spx_clean.to_csv(data_path + 'sp500_clean.csv')


# NFCI
nfci = pd.read_csv('https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23ebf3fb&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1320&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=NFCI&scale=left&cosd=2000-01-01&coed=2026-05-01&line_color=%230073e6&link_values=false&line_style=solid&mark_type=none&mw=3&lw=3&ost=-99999&oet=99999&mma=0&fml=a&fq=Weekly%2C%20Ending%20Friday&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2026-05-11&revision_date=2026-05-11&nd=1971-01-08')
nfci.to_csv(data_path + 'NFCI.csv')

# STLFSI4
stlfsi4 = pd.read_csv('https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23ebf3fb&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1320&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=STLFSI4&scale=left&cosd=2000-01-01&coed=2026-05-01&line_color=%230073e6&link_values=false&line_style=solid&mark_type=none&mw=3&lw=3&ost=-99999&oet=99999&mma=0&fml=a&fq=Weekly%2C%20Ending%20Friday&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2026-05-11&revision_date=2026-05-11&nd=1993-12-31')
stlfsi4.to_csv(data_path + 'STLFSI4.csv')


# Generate OFR and VIX episodes based on rolling 90th and 95th percentile, 
# and save the episodes in the data folder as table1_OFR_rolling90_episodes.csv, 
# table2_VIX_rolling90_episodes.csv, table3_OFR_and_VIX_rolling90_episodes.csv, 
# table1_OFR_rolling95_episodes.csv, table2_VIX_rolling95_episodes.csv, table3_OFR_and_VIX_rolling95_episodes.csv
data_path = "../data/"

vix_path = "../data/4columns_na_dropped.csv"
ofr_path = "../data/labels.csv"
spx_path = "../data/sp500_clean.csv"

date_col = "date"

vix_col = "VIX"
ofr_col = "OFR FSI"
price_col = "spx"

vix = pd.read_csv(vix_path)
ofr = pd.read_csv(ofr_path)
spx = pd.read_csv(spx_path)

vix = vix.rename(columns={vix.columns[0]: "date"})
ofr = ofr.rename(columns={"Date": "date"})

vix[date_col] = pd.to_datetime(vix[date_col])
ofr[date_col] = pd.to_datetime(ofr[date_col])
spx[date_col] = pd.to_datetime(spx[date_col])

merged = (
    vix.merge(ofr, on=date_col, how="inner")
       .sort_values(date_col)
       .reset_index(drop=True)
)
df = (
    merged.merge(spx, on=date_col, how="inner")
          .sort_values(date_col)
          .reset_index(drop=True)
)

for c in [vix_col, ofr_col]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=[vix_col, ofr_col]).reset_index(drop=True)

q = 0.9
rolling_window = 252
min_periods = 126


def rolling_q(s, window=rolling_window, q=q, min_periods=min_periods):
    return s.rolling(window, min_periods=min_periods).quantile(q).shift(1)


def make_episodes(df, flag_col, date_col="date", peak_col=None):
    d = df.copy()
    flag = d[flag_col].fillna(False).astype(bool)

    if not flag.any():
        return pd.DataFrame(columns=["onset", "end", "n_days", "peak_date", "peak_value"])

    group_id = (flag != flag.shift(fill_value=False)).cumsum()
    rows = []

    for _, g in d[flag].groupby(group_id[flag]):
        onset = g[date_col].iloc[0]
        end = g[date_col].iloc[-1]

        if peak_col is not None:
            idx = g[peak_col].idxmax()
            peak_date = d.loc[idx, date_col]
            peak_value = d.loc[idx, peak_col]
        else:
            peak_date = onset
            peak_value = np.nan

        rows.append({
            "onset": onset,
            "end": end,
            "n_days": len(g),
            "peak_date": peak_date,
            "peak_value": peak_value
        })

    return pd.DataFrame(rows)


# =====================
# 1. rolling OFR > 90th
# =====================

df["ofr_roll_q90"] = rolling_q(
    df[ofr_col],
    window=rolling_window,
    q=q,
    min_periods=min_periods
)

df["OFR_gt_rolling90"] = df[ofr_col] >= df["ofr_roll_q90"]

table1_ofr_rolling90 = make_episodes(
    df,
    flag_col="OFR_gt_rolling90",
    date_col="date",
    peak_col=ofr_col
)


# =====================
# 2. rolling VIX > 90th
# =====================

df["vix_roll_q90"] = rolling_q(
    df[vix_col],
    window=rolling_window,
    q=q,
    min_periods=min_periods
)

df["VIX_gt_rolling90"] = df[vix_col] >= df["vix_roll_q90"]

table2_vix_rolling90 = make_episodes(
    df,
    flag_col="VIX_gt_rolling90",
    date_col="date",
    peak_col=vix_col
)


# =====================
# 3. rolling OFR & rolling VIX > 90th
# =====================

confirm_window = 5

df["OFR_recent90"] = (
    df["OFR_gt_rolling90"]
    .fillna(False)
    .astype(int)
    .rolling(confirm_window, min_periods=1)
    .max()
    .astype(bool)
)

df["VIX_recent90"] = (
    df["VIX_gt_rolling90"]
    .fillna(False)
    .astype(int)
    .rolling(confirm_window, min_periods=1)
    .max()
    .astype(bool)
)

df["OFR_and_VIX_gt_rolling90"] = (
    df["OFR_recent90"] & df["VIX_recent90"]
)

table3_ofr_vix_rolling90 = make_episodes(
    df,
    flag_col="OFR_and_VIX_gt_rolling90",
    date_col="date",
)

table1_ofr_rolling90.to_csv(data_path + "table1_OFR_rolling90_episodes.csv", index=False)
table2_vix_rolling90.to_csv(data_path + "table2_VIX_rolling90_episodes.csv", index=False)
table3_ofr_vix_rolling90.to_csv(data_path + "table3_OFR_and_VIX_rolling90_episodes.csv", index=False)

# Now try q=0.95
q = 0.95
rolling_window = 252
min_periods = 126

# =====================
# 1. rolling OFR > 95th
# =====================

df["ofr_roll_q95"] = rolling_q(
    df[ofr_col],
    window=rolling_window,
    q=q,
    min_periods=min_periods
)

df["OFR_gt_rolling95"] = df[ofr_col] >= df["ofr_roll_q95"]

table1_ofr_rolling95 = make_episodes(
    df,
    flag_col="OFR_gt_rolling95",
    date_col="date",
    peak_col=ofr_col
)


# =====================
# 2. rolling VIX > 95th
# =====================

df["vix_roll_q95"] = rolling_q(
    df[vix_col],
    window=rolling_window,
    q=q,
    min_periods=min_periods
)

df["VIX_gt_rolling95"] = df[vix_col] >= df["vix_roll_q95"]

table2_vix_rolling95 = make_episodes(
    df,
    flag_col="VIX_gt_rolling95",
    date_col="date",
    peak_col=vix_col
)


# =====================
# 3. rolling OFR & rolling VIX > 95th
# =====================

confirm_window = 5

df["OFR_recent95"] = (
    df["OFR_gt_rolling95"]
    .fillna(False)
    .astype(int)
    .rolling(confirm_window, min_periods=1)
    .max()
    .astype(bool)
)

df["VIX_recent95"] = (
    df["VIX_gt_rolling95"]
    .fillna(False)
    .astype(int)
    .rolling(confirm_window, min_periods=1)
    .max()
    .astype(bool)
)

df["OFR_and_VIX_gt_rolling95"] = (
    df["OFR_recent95"] & df["VIX_recent95"]
)

table3_ofr_vix_rolling95 = make_episodes(
    df,
    flag_col="OFR_and_VIX_gt_rolling95",
    date_col="date",
)

table1_ofr_rolling95.to_csv(data_path + "table1_OFR_rolling95_episodes.csv", index=False)
table2_vix_rolling95.to_csv(data_path + "table2_VIX_rolling95_episodes.csv", index=False)
table3_ofr_vix_rolling95.to_csv(data_path + "table3_OFR_and_VIX_rolling95_episodes.csv", index=False)