"""Market-neutral alpha research on ~100 large US stocks (2010-2025).

Three classic signals, each turned into a dollar-neutral daily portfolio, then combined.
Signals are computed with data up to day t's close and earn day t+1's return (no lookahead).
Train = 2010-2017 (only place anything is fitted), Test = 2018-2025 (out-of-sample).

Run:  python analysis/alpha_research.py   ->  prints results, writes outputs/
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)
TRAIN_END = "2017-12-31"
COST_BPS = 5  # one-way cost per unit traded, in basis points


def zscore(df):
    """Cross-sectional z-score each day (across stocks)."""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


def to_weights(signal):
    """Dollar-neutral weights: demeaned z-score scaled to gross exposure 1 (sum |w| = 1)."""
    z = zscore(signal)
    return z.div(z.abs().sum(axis=1), axis=0).fillna(0.0)


def rank_ic(signal, fwd):
    """Daily Spearman correlation between today's signal and tomorrow's return."""
    s, f = signal.rank(axis=1), fwd.rank(axis=1)
    ok = s.notna() & f.notna()
    s, f = s.where(ok), f.where(ok)
    s = s.sub(s.mean(axis=1), axis=0)
    f = f.sub(f.mean(axis=1), axis=0)
    return (s * f).sum(axis=1) / np.sqrt((s ** 2).sum(axis=1) * (f ** 2).sum(axis=1))


def stats(pnl):
    pnl = pnl.dropna()
    ann_ret = pnl.mean() * 252
    ann_vol = pnl.std() * np.sqrt(252)
    cum = pnl.cumsum()
    return {"ann_return_pct": round(100 * ann_ret, 2), "ann_vol_pct": round(100 * ann_vol, 2),
            "sharpe": round(ann_ret / ann_vol, 2) if ann_vol > 0 else None,
            "max_drawdown_pct": round(100 * (cum - cum.cummax()).min(), 2)}


def backtest(w, fwd):
    gross = (w * fwd).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1)
    net = gross - turnover * COST_BPS / 1e4
    return gross, net, turnover


def main():
    px = pd.read_csv(os.path.join(ROOT, "data", "prices.csv"), index_col=0, parse_dates=True)
    px = px.ffill(limit=3)
    ret = px.pct_change(fill_method=None)
    fwd = ret.shift(-1)  # return earned by holding from today's close to tomorrow's close
    mkt = ret.mean(axis=1)  # equal-weight market, for a beta check

    signals = {
        "reversal_5d": -(px / px.shift(5) - 1),                 # last week's losers bounce back
        "momentum_12_1": px.shift(21) / px.shift(252) - 1,      # past-year winners keep winning
        "low_vol_60d": -ret.rolling(60).std(),                  # calmer stocks earn more per unit risk
    }
    # equal-weight combination of z-scores (nothing fitted)
    signals["combo_equal"] = sum(zscore(s) for s in signals.values()) / 3

    # IC-weighted combination: weights = each signal's mean rank IC in the TRAIN period only
    train_ic = {k: rank_ic(signals[k], fwd).loc[:TRAIN_END].mean()
                for k in ("reversal_5d", "momentum_12_1", "low_vol_60d")}
    tot = sum(abs(v) for v in train_ic.values())
    ic_w = {k: v / tot for k, v in train_ic.items()}
    signals["combo_ic_weighted"] = sum(ic_w[k] * zscore(signals[k]) for k in ic_w)

    start = px.index[260]  # need ~1 year of history for momentum
    results, curves = {}, {}
    for name, sig in signals.items():
        w = to_weights(sig).loc[start:]
        f = fwd.loc[start:]
        gross, net, to = backtest(w, f)
        ic = rank_ic(sig.loc[start:], f)
        r = {}
        for period, sl in (("train", slice(None, TRAIN_END)), ("test", slice("2018-01-01", None))):
            g, n, t, i = gross.loc[sl], net.loc[sl], to.loc[sl], ic.loc[sl].dropna()
            r[period] = {
                "gross": stats(g), "net_5bps": stats(n),
                "avg_daily_turnover_pct": round(100 * t.mean(), 1),
                "mean_rank_ic": round(i.mean(), 4),
                "ic_t_stat": round(i.mean() / i.std() * np.sqrt(len(i)), 2),
                # cost (bps, one-way) at which gross profit is fully eaten by trading
                "breakeven_cost_bps": round(1e4 * g.mean() / t.mean(), 1) if t.mean() > 0 else None,
                "corr_with_market": round(g.corr(mkt.shift(-1).loc[g.index]), 2),
            }
        results[name] = r
        curves[name] = net
    results["ic_weights_from_train"] = {k: round(v, 3) for k, v in ic_w.items()}
    results["universe"] = {"stocks": int(px.shape[1]), "start": str(start.date()),
                           "end": str(px.index[-2].date()), "train_end": TRAIN_END,
                           "cost_bps_one_way": COST_BPS}

    # how correlated are the three base strategies (diversification)
    base = pd.DataFrame({k: backtest(to_weights(signals[k]).loc[start:], fwd.loc[start:])[0]
                         for k in ("reversal_5d", "momentum_12_1", "low_vol_60d")})
    results["pnl_correlations"] = base.corr().round(2).to_dict()

    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    plt.figure(figsize=(8, 4))
    for name, c in curves.items():
        plt.plot((100 * c.cumsum()), label=name, lw=1)
    plt.axvline(pd.Timestamp("2018-01-01"), color="black", ls="--", lw=1)
    plt.text(pd.Timestamp("2018-03-01"), plt.ylim()[1] * 0.9, "out-of-sample", fontsize=8)
    plt.ylabel("cumulative net return (%), 5 bps costs")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "cumulative_pnl.png"), dpi=150)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
