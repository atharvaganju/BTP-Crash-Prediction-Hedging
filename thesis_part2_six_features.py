"""
Thesis part 2: the six fragility features.

  1. Markov-switching filtered probability of the high-volatility state (S&P 500 fund returns)
  2. Ratio of 30-day to 3-month implied volatility (volatility term structure)
  3. 20-day change in log(high-yield bond fund / investment-grade bond fund)   (credit risk appetite)
  4. Ten-year minus two-year Treasury yield slope                              (growth and policy expectations)
  5. 20-day change in the ten-year real (inflation-protected) Treasury yield   (discount-rate shock)
  6. Average pairwise 60-day correlation of the three asset classes            (diversification breakdown)

Look-ahead protection:
  * Markov-switching parameters are re-estimated on an expanding window; each day's probability uses
    only parameters estimated from earlier data and returns up to that day (filtered, never smoothed).
  * Federal Reserve series are lagged by one trading day because they are published after the close.

Run:  pip install yfinance pandas numpy pyarrow statsmodels
Needs btp_part1_data_labels.py in the same folder. UNTESTED: my sandbox cannot reach Yahoo or the Federal Reserve data service.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.stattools import adfuller

import btp_part1_data_labels as part_one

# Data timing is exactly as in Part 1: download starts 2015-06-01, analysis window starts 2016-10-01.
# Nothing before the ten-year window is added.
MINIMUM_TRAINING_DAYS = 252      # one year of trading days before the first fit (lands just before the window opens)
REFIT_EVERY = 63                 # re-estimate parameters about once a quarter
CORRELATION_WINDOW = 60
CHANGE_WINDOW = 20


# ---------- feature 4 and 5 data ----------
def load_federal_reserve_series(series_identifier: str, target_index: pd.DatetimeIndex) -> pd.Series:
    """Download a free Federal Reserve Economic Data series, align to trading days, lag one day."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_identifier}"
    frame = pd.read_csv(url, index_col=0, parse_dates=True, na_values=".")
    series = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
    series = series.reindex(series.index.union(target_index)).ffill().reindex(target_index)
    return series.shift(1)       # published after the close, so usable only from the next day


# ---------- feature 1 ----------
def markov_switching_probability(returns_percent: pd.Series) -> pd.Series:
    """Filtered probability of the high-volatility state, with expanding-window re-estimation."""
    probability = pd.Series(np.nan, index=returns_percent.index, name="markov_switching_probability")
    previous_parameters = None

    for start in range(MINIMUM_TRAINING_DAYS, len(returns_percent), REFIT_EVERY):
        end = min(start + REFIT_EVERY, len(returns_percent))
        training = returns_percent.iloc[:start]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = MarkovRegression(training, k_regimes=2, trend="c",
                                          switching_variance=True).fit(search_reps=10)
            previous_parameters = fitted.params
        except Exception as error:   # fall back to the last good parameters rather than abort
            print(f"fit failed at {returns_percent.index[start].date()}: {error}")
            if previous_parameters is None:
                continue

        # Apply the fixed parameters to all data up to `end`; the filter only looks backward.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filtered = MarkovRegression(returns_percent.iloc[:end], k_regimes=2, trend="c",
                                        switching_variance=True).filter(previous_parameters)

        # The state labels can swap between fits, so identify the high-volatility state by its variance.
        variances = [previous_parameters["sigma2[0]"], previous_parameters["sigma2[1]"]]
        high_volatility_state = int(np.argmax(variances))
        probability.iloc[start:end] = np.asarray(
            filtered.filtered_marginal_probabilities)[start:end, high_volatility_state]

    return probability


# ---------- feature 6 ----------
def average_pairwise_correlation(log_returns: pd.DataFrame, window: int) -> pd.Series:
    pairs = [("SPY", "TLT"), ("SPY", "GLD"), ("TLT", "GLD")]
    correlations = [log_returns[a].rolling(window).corr(log_returns[b]) for a, b in pairs]
    return pd.concat(correlations, axis=1).mean(axis=1).rename("average_pairwise_correlation")


# ---------- assemble ----------
def build_features(close: pd.DataFrame) -> pd.DataFrame:
    log_returns = np.log(close[part_one.ASSETS]).diff()
    returns_percent = 100 * log_returns["SPY"].dropna()      # percent scale helps the optimizer converge

    features = pd.DataFrame(index=close.index)
    features["markov_switching_probability"] = markov_switching_probability(returns_percent)
    features["volatility_term_structure"] = close["^VIX"] / close["^VIX3M"]
    features["credit_appetite_change"] = np.log(close["HYG"] / close["LQD"]).diff(CHANGE_WINDOW)
    features["yield_curve_slope"] = load_federal_reserve_series("T10Y2Y", close.index)
    features["real_yield_change"] = load_federal_reserve_series("DFII10", close.index).diff(CHANGE_WINDOW)
    features["average_pairwise_correlation"] = average_pairwise_correlation(log_returns, CORRELATION_WINDOW)
    return features


def diagnostics(features: pd.DataFrame) -> None:
    print("\n=== Feature correlation matrix (investigate any pair above 0.8 in absolute value) ===")
    print(features.corr().round(2).to_string())
    print("\n=== Augmented Dickey-Fuller p-values (want below 0.05 for stationarity) ===")
    for column in features.columns:
        p_value = adfuller(features[column].dropna())[1]
        print(f"{column:<32} {p_value:.4f}")
    print("\n=== Missing values per feature inside the analysis window ===")
    print(features.isna().sum().to_string())


if __name__ == "__main__":
    close, _volume = part_one.load_market_data()
    features = build_features(close).loc[part_one.ANALYSIS_START:]
    diagnostics(features)
    features.dropna().to_parquet("features.parquet")
    print(f"\nSaved features.parquet with {len(features.dropna())} complete rows.")