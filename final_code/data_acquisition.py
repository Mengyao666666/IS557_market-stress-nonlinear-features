'''
This code is responsible for acquiring the necessary data for our analysis. 
We will be using the yfinance library and url from FRED to fetch financial data and pandas to handle data manipulation. 
Since FRED Data does not allow reproduce or store data, we leave this script to be run by users to acquire the data and save it in the data folder.

AI Usage: This code was generated with the assistance of AI, which helped in structuring the data acquisition process and ensuring that the correct libraries and methods were used to fetch and store the data efficiently.
'''
import pandas as pd
import yfinance as yf

# Acquire Spread 10 year to 2 year data from FRED, in CSV format that already provided
df_spread = pd.read_csv('https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23ebf3fb&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1320&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=T10Y2Y&scale=left&cosd=2000-01-01&coed=2026-03-09&line_color=%230073e6&link_values=false&line_style=solid&mark_type=none&mw=3&lw=3&ost=-99999&oet=99999&mma=0&fml=a&fq=Daily&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2026-05-11&revision_date=2026-05-11&nd=1976-06-01')
# Save the data to a CSV file
df_spread.to_csv('data/spread_10y2y.csv', index=False)

# This is very unfortunate, but FRED does not provide High Yeild OAS data older than 3 years.
# Please have contact with FRED to acquire the data, and save it in the data folder as hy_oas.csv
df_oas = pd.read_csv("data/hy_oas.csv")

# Acquire SKEW data from CBOE, from yfinance library, and save it in the data folder as CBOE_SKEW_Full_2000_2026.csv
df_skew = yf.download("^SKEW", start="2000-01-01", end="2026-03-09")
df_skew.to_csv("data/CBOE_SKEW_Full_2000_2026.csv")

# Acquire VIX data from CBOE, from yfinance library, and save it in the data folder as VIX.csv
df_vix = yf.download("^VIX", start="2000-01-01", end="2026-03-09")
df_vix.to_csv("data/VIX.csv")

# Data Cleaning to get 4columns_na_dropped.csv
dataframes = {
    "SKEW": df_skew,
    "VIX": df_vix,
    "10y-2y Spread": df_spread,
    "HY OAS": df_oas
}

data_final = pd.concat([
    df_skew['Close'].rename('SKEW'),
    df_vix['Close'].rename('VIX'),
    df_spread['T10Y2Y'].rename('Spread'),
    df_oas['BAMLH0A0HYM2'].rename('OAS')
], axis=1)

data_model = data_final.dropna().copy()
data_model.to_csv("data/4columns_na_dropped.csv")