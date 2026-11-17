"""Version 2: fix the two problems found in version 1 (alpha_research.py).

Problem 1: "dollar-neutral" portfolios still had big market exposure (low-vol: -0.6 corr with market).
  Fix: beta-neutralise. Each day, remove the part of the weights explained by [1, beta]
  (rolling 1-year beta of each stock to the equal-weight market), so the book is both
  dollar-neutral and beta-neutral.
Problem 2: reversal traded ~63% of the book per day, so costs ate the edge.
  Fix: trade gradually towards the target: w_t = lam * target_t + (1 - lam) * w_{t-1}.
  lam is chosen per signal by best NET Sharpe on the TRAIN period only (2011-2017).

Honesty note: the test period (2018-2025) was looked at once in version 1 before these fixes
were designed, so this test is not perfectly untouched. The fixes come from v1 diagnostics
(market correlation, turnover), not from tuning on test returns.

Run:  python analysis/alpha_v2.py   ->  prints results, writes outputs/results_v2.json and chart
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alpha_research import COST_BPS, TRAIN_END, rank_ic, stats, zscore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
LAMBDAS = [1.0, 0.5, 0.25, 0.1, 0.05]


def beta_neutral_weights(signal, beta):
    z = zscore(signal)
    out = np.zeros(z.shape)
    zv, bv = z.to_numpy(), beta.to_numpy()
    for i in range(len(z)):
        s, b = zv[i], bv[i]
        ok = ~np.isnan(s) & ~np.isnan(b)
        if ok.sum() < 20:
            continue
        X = np.column_stack([np.ones(ok.sum()), b[ok]])
        coef, *_ = np.linalg.lstsq(X, s[ok], rcond=None)
        resid = s[ok] - X @ coef          # signal with no dollar or beta exposure
        g = np.abs(resid).sum()
        if g > 0:
            out[i, ok] = resid / g        # gross exposure 1
    return pd.DataFrame(out, index=z.index, columns=z.columns)


def run(w, fwd):
    gross = (w * fwd).sum(axis=1)
    to = w.diff().abs().sum(axis=1)
    return gross, gross - to * COST_BPS / 1e4, to


def main():
    px = pd.read_csv(os.path.join(ROOT, "data", "prices.csv"), index_col=0, parse_dates=True)
    px = px.ffill(limit=3)
    ret = px.pct_change(fill_method=None)
    fwd = ret.shift(-1)
    mkt = ret.mean(axis=1)
    beta = ret.rolling(252, min_periods=200).cov(mkt).div(mkt.rolling(252, min_periods=200).var(), axis=0)

    sig = {"reversal_5d": -(px / px.shift(5) - 1),
           "momentum_12_1": px.shift(21) / px.shift(252) - 1,
           "low_vol_60d": -ret.rolling(60).std()}
    train_ic = {k: rank_ic(s, fwd).loc[:TRAIN_END].mean() for k, s in sig.items()}
    tot = sum(abs(v) for v in train_ic.values())
    sig["combo_ic_weighted"] = sum((train_ic[k] / tot) * zscore(sig[k]) for k in list(train_ic))

    start = px.index[260]
    f = fwd.loc[start:]
    results, curves = {}, {}
    for name, s in sig.items():
        target = beta_neutral_weights(s.loc[start:], beta.loc[start:])
        # pick lambda on TRAIN net Sharpe only
        best = None
        for lam in LAMBDAS:
            w = target.ewm(alpha=lam, adjust=False).mean()
            _, net, _ = run(w, f)
            sh = stats(net.loc[:TRAIN_END])["sharpe"]
            if best is None or sh > best[1]:
                best = (lam, sh)
        lam = best[0]
        w = target.ewm(alpha=lam, adjust=False).mean()
        gross, net, to = run(w, f)
        r = {"lambda_chosen_on_train": lam}
        for period, sl in (("train", slice(None, TRAIN_END)), ("test", slice("2018-01-01", None))):
            g, n, t = gross.loc[sl], net.loc[sl], to.loc[sl]
            r[period] = {"gross": stats(g), "net_5bps": stats(n),
                         "avg_daily_turnover_pct": round(100 * t.mean(), 1),
                         "breakeven_cost_bps": round(1e4 * g.mean() / t.mean(), 1),
                         "corr_with_market": round(g.corr(mkt.shift(-1).loc[g.index]), 2)}
        results[name] = r
        curves[name] = net

    with open(os.path.join(OUT, "results_v2.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    plt.figure(figsize=(8, 4))
    for name, c in curves.items():
        plt.plot(100 * c.cumsum(), label=name, lw=1)
    plt.axvline(pd.Timestamp("2018-01-01"), color="black", ls="--", lw=1)
    plt.ylabel("cumulative net return (%), 5 bps costs")
    plt.title("v2: beta-neutral, smoothed trading")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "cumulative_pnl_v2.png"), dpi=150)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
