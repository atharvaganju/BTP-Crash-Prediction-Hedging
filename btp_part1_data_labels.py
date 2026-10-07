"""
BTP Part 1: data pipeline + mechanical crash labelling (SPY, TLT, GLD).

Label definition (no look-ahead in the threshold):
    y_t = 1  if  min_{t < tau <= t+k} (P_tau / P_t - 1)  <  -theta_t
    theta_t = fixed  OR  z * sigma_t * sqrt(k)   (sigma_t = trailing 252d daily log-ret std, known at t)
Last k rows have no full forward window -> label is NaN (NOT 0). Drop them before training.

Run:  pip install yfinance pandas numpy pyarrow   (UNTESTED here: no network to Yahoo in my sandbox)
"""
import numpy as np
import pandas as pd
import yfinance as yf

# ---------- config ----------
ANALYSIS_START = "2016-10-01"      # 10-year window ending ~now
DATA_START = "2015-06-01"          # >1y burn-in for trailing-1y stats
ASSETS = ["SPY", "TLT", "GLD"]
AUX = ["^VIX", "^VIX3M", "HYG", "LQD"]
K = 20                             # forward window (trading days)
GAP = 20                           # merge flagged days closer than this into one episode


# ---------- data ----------
def load_market_data():
    df = yf.download(ASSETS + AUX, start=DATA_START, auto_adjust=True, progress=False)
    close, volume = df["Close"], df["Volume"]
    core = close[ASSETS].dropna().index
    close = close.loc[core].copy()
    close[AUX] = close[AUX].ffill()          # index series have different holidays
    volume = volume.loc[core, ASSETS]
    return close, volume


# ---------- labelling ----------
def forward_drawdown(p: pd.Series, k: int) -> pd.Series:
    """min over (t, t+k] of P_tau/P_t - 1, capped at 0. NaN for last k rows."""
    fwd_min = p.rolling(k).min().shift(-k)   # value at t = min(P_{t+1..t+k})
    return (fwd_min / p - 1).clip(upper=0)


def theta_series(p: pd.Series, k: int, mode="vol", fixed=0.15, z=2.0) -> pd.Series:
    if mode == "fixed":
        return pd.Series(fixed, index=p.index)
    sigma = np.log(p).diff().rolling(252).std()      # uses data up to t only
    return z * sigma * np.sqrt(k)


def make_labels(p: pd.Series, k=K, mode="vol", fixed=0.15, z=2.0) -> pd.Series:
    dd = forward_drawdown(p, k)
    th = theta_series(p, k, mode, fixed, z)
    y = (dd < -th).astype(float)
    y[dd.isna() | th.isna()] = np.nan
    return y


def vol_spike_flag(p: pd.Series, mult=2.0) -> pd.Series:
    """Contemporaneous cross-check (NOT forward-looking): 20d realised vol > mult x trailing-1y mean."""
    rv = np.log(p).diff().rolling(20).std() * np.sqrt(252)
    return (rv > mult * rv.rolling(252).mean()).astype(float).where(rv.rolling(252).mean().notna())


def episodes(y: pd.Series, p: pd.Series, k=K, gap=GAP) -> pd.DataFrame:
    """Merge consecutive flagged days into discrete crash episodes."""
    idx = np.flatnonzero(y.fillna(0).values == 1)
    if len(idx) == 0:
        return pd.DataFrame()
    out = []
    for g in np.split(idx, np.where(np.diff(idx) > gap)[0] + 1):
        s, e = g[0], g[-1]
        win = p.iloc[s: min(e + k, len(p) - 1) + 1]
        out.append(dict(first_flag=p.index[s].date(), last_flag=p.index[e].date(),
                        flagged_days=len(g), trough=win.idxmin().date(),
                        dd_to_trough=round(win.min() / p.iloc[s] - 1, 4)))
    return pd.DataFrame(out)


def sensitivity_table(p: pd.Series) -> pd.DataFrame:
    cfgs = [("fixed 10%", dict(mode="fixed", fixed=0.10)),
            ("fixed 15%", dict(mode="fixed", fixed=0.15)),
            ("vol z=1.5", dict(mode="vol", z=1.5)),
            ("vol z=2.0", dict(mode="vol", z=2.0)),
            ("vol z=2.5", dict(mode="vol", z=2.5))]
    rows = []
    for name, kw in cfgs:
        y = make_labels(p, **kw).loc[ANALYSIS_START:]
        rows.append(dict(rule=name, episodes=len(episodes(y, p.loc[y.index])),
                         pos_rate=round(y.mean(), 4), n_obs=int(y.notna().sum())))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    close, volume = load_market_data()
    spy = close["SPY"]

    print("\n=== Sensitivity of crash-episode count to the rule (SPY) ===")
    print(sensitivity_table(spy).to_string(index=False))

    # default rule: vol-scaled, z=2
    y = make_labels(spy, mode="vol", z=2.0)
    ep = episodes(y.loc[ANALYSIS_START:], spy.loc[ANALYSIS_START:])
    print("\n=== Episodes under default rule (vol z=2.0) -> map these to named events by hand ===")
    print(ep.to_string(index=False))

    vs = vol_spike_flag(spy)
    print("\nVol-spike days (cross-check only):", int(vs.loc[ANALYSIS_START:].sum()))

    out = pd.DataFrame({"y_crash": y, "fwd_dd": forward_drawdown(spy, K), "volspike": vs})
    close.loc[ANALYSIS_START:].to_parquet("prices.parquet")
    volume.loc[ANALYSIS_START:].to_parquet("volume.parquet")
    out.loc[ANALYSIS_START:].to_parquet("labels_spy.parquet")
    print("\nSaved prices.parquet, volume.parquet, labels_spy.parquet")