# Financial Stress Detection with Nonlinear Features
 
**IS557 Final Project — University of Illinois Urbana-Champaign**  
Mengyao Wang · Changho Jung · Chunyu Liu

## Project Goal

The goal of this project is to evaluate whether nonlinear structural features can improve financial stress detection, especially for stress events related to:

- Equity volatility stress
- Credit-market stress
- Option-market tail-risk pricing
- Broad systemic financial stress

We focus on two main research questions:

1. Do temporal scaling features and phase-space geometric features provide complementary information within the same financial series?
2. Do cross-market nonlinear features provide additional predictive information beyond the target market’s own variables?
## Data
 
Four market-based stress proxies: **VIX** (equity volatility), **OAS** (credit stress), **SKEW** (tail-risk pricing), **OFR FSI** (broad systemic stress). Daily frequency, sourced from CBOE and the Office of Financial Research.
 
High-stress events defined via 90th/95th percentile thresholds of VIX and OFR FSI, and 95th percentile thresholds of OAS and SKEW.
 
## Methodology

The project uses a two-stage evaluation framework.

### Stage 1: Change Point Detection

We apply change point detection to test whether nonlinear features identify structural breaks around financial stress windows. A placebo-style random benchmark is used to check whether detected change points align with stress events more often than expected by chance.

### Stage 2: Walk-Forward Prediction

We use logistic regression in a recursive walk-forward setting to evaluate out-of-sample prediction. This tests whether nonlinear features improve future stress detection beyond conventional financial indicators.

## Main Findings

The results suggest that nonlinear features are useful, but their value depends on the type of stress event.

- **MFDFA-based scaling features** are more useful for broad systemic stress, especially OFR-related stress windows.
- **Takens-based correlation dimension** provides stronger signals for volatility-related market reorganization, especially VIX stress events.
- In the credit market, both temporal scaling and phase-space geometry provide complementary information.
- Cross-market features are important. For example, credit-market geometry and option-market tail-risk features improve prediction of VIX and OFR stress events.
- SKEW behaves differently from VIX and OAS. Nonlinear features are less effective at predicting high-SKEW events directly, but they help identify stable non-stress periods.

Overall, nonlinear structural features appear to provide useful supplementary signals for financial stress monitoring.

## Limitations and Future Work

One limitation is that the predictive value of nonlinear features appears to be regime-dependent. The features perform less consistently around the 2008 crisis and after 2022, which may reflect different macro-financial regimes.

Future work could evaluate the framework separately across different periods, such as:

- Pre- and post-Global Financial Crisis periods
- Post-GFC low-interest-rate period
- Post-2022 high-inflation and rate-hike period

This may help determine whether nonlinear structural features are stable early-warning signals or whether their usefulness changes across market regimes.

### How to Reproduce work?
#### Datasets
- Datasets from Federal Reserve Bank of St. Louis, FRED
1. spread_10y2y.csv ([Link](https://fred.stlouisfed.org/series/T10Y2Y))
2. hy_oas.csv (Very very unfortunatly, FRED no longer provide data the last more than 3 years since April of 2026. FRED also does not allow reproduce or redistribute the data by Copyright, 2023, ICE Data Indices. Please [contact](https://fred.stlouisfed.org/contactus/) FRED.)
- Datasets from yfinance library
3. CBOE_SKEW_Full_2000_2026.csv
4. VIX.csv

#### Reproduction Script Order
1. data_acquisition.py
2. mfdfa_features.py
3. takens_embedding.py
4. cpd_pipeline.py
5. logreg_pipeline.py

- In command shell main directory

```
python final_code/data_acquisition.py
python final_code/mfdfa_features.py
python final_code/takens_embedding.py
python final_code/cpd_pipeline.py
python final_code/logreg_pipeline.py
```


#### AI Usage
- Our projects done by Python codes, and they are assisted by ChatGPT and GitHub Copilot
