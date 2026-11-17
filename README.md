# Market-Neutral Alpha Research on US Equities

I tested three classic stock-selection signals on 104 large US stocks from 2011 to 2025. Each one is turned into a
daily long-short portfolio and checked out-of-sample after trading costs. Then I fixed the two problems the first
version exposed.

**Data:** daily adjusted close prices from Yahoo Finance (`analysis/fetch_prices.py`), 104 stocks, 2010–2025.
**Run:** `python analysis/fetch_prices.py`, then `python analysis/alpha_research.py` (v1), then `python analysis/alpha_v2.py` (v2).
Outputs land in `outputs/` (`results.json`, `results_v2.json`, and the cumulative PnL charts).

## Setup

| Item | Choice |
|---|---|
| Signals | **5-day reversal** (last week's losers bounce back), **12-1 momentum** (the past year's winners, skipping the last month, keep winning), **60-day low volatility** (calmer stocks earn more per unit of risk) |
| Portfolio | Each day: cross-sectional z-score → demean → scale to gross exposure 1. That's dollar-neutral: $1 long and $1 short in total |
| Timing | The signal uses data up to day *t*'s close and earns day *t+1*'s return, so there's no lookahead |
| Split | **Train 2011–2017** (the only period where anything is fitted). **Test 2018–2025** (out-of-sample) |
| Costs | 5 basis points one-way per unit traded |
| Combination | Each signal's weight is its mean rank IC in the train period: reversal 0.46, momentum 0.43, low-vol 0.11 |

**Metrics.** Rank IC is the daily Spearman correlation between the signal and the next day's return. Sharpe is annual
return ÷ annual volatility, using √252. Turnover is the share of the book traded per day. Break-even cost is the
cost level at which gross profit hits zero.

## Version 1: plain dollar-neutral portfolios

| Signal (test 2018–2025) | Rank IC (t-stat) | Gross Sharpe | Net Sharpe (5 bps) | Daily turnover | Corr. with market |
|---|---|---|---|---|---|
| Reversal 5d | 0.008 (1.4) | 0.05 | −0.77 | 63% | 0.25 |
| Momentum 12-1 | **0.018 (2.9)** | 0.19 | 0.08 | 9% | −0.16 |
| Low vol 60d | −0.004 (−0.5) | −0.59 | −0.65 | 6% | **−0.56** |
| IC-weighted combo | 0.016 (2.8) | 0.01 | **−0.62** | 46% | −0.01 |

What went wrong:
1. **Costs ate the reversal signal.** In the train period it had a real edge (IC t-stat 3.7, gross Sharpe 0.81), but
   trading 63% of the book every day means 2.8 bps of cost wipes it out. It also weakened after 2018.
2. **Dollar-neutral isn't market-neutral.** Low-vol was −0.6 correlated with the market. Low-volatility stocks have low
   beta, so long low-vol and short high-vol is effectively short the market, and that loses in a bull market.
3. **Momentum was the only signal that held up.** Its IC was nearly identical in train and test (0.017 vs 0.018, t = 2.9
   in both).

## Version 2: beta-neutral, smoothed trading

- **Beta-neutralise:** each day I regress the signal on [1, beta] across stocks and keep the residual. Beta is each
  stock's rolling 1-year beta to the equal-weight market. The portfolio then has zero net dollars and zero net beta.
- **Trade gradually:** `w_t = λ·target_t + (1−λ)·w_{t−1}`. λ is picked from {1, 0.5, 0.25, 0.1, 0.05} by the best
  **net Sharpe in the train period only**. It came out at 0.1 for reversal, momentum and the combo, and 0.5 for low-vol.

| Signal (test 2018–2025) | Gross Sharpe | Net Sharpe (5 bps) | Daily turnover | Break-even cost | Corr. with market |
|---|---|---|---|---|---|
| Reversal 5d | 0.29 | −0.03 | 9% | 4.5 bps | 0.30 |
| **Momentum 12-1** | 0.47 | **0.44** | 2% | 71 bps | −0.09 |
| Low vol 60d | −0.26 | −0.35 | 5% | n/a | −0.17 |
| **IC-weighted combo** | 0.56 | **0.38** | **7%** (from 46%) | 16 bps | **0.03** |

The combined signal's out-of-sample net Sharpe went from **−0.62 to +0.38**. Its turnover fell 46% → 7% a day, and its
market correlation is now about zero.

## Caveats (I'd raise these myself)

- **Survivorship bias.** The universe is companies that are large *today*. Stocks that failed since 2010 are missing,
  which flatters long-only returns and can bias these signals too.
- **The test set isn't perfectly clean.** I looked at test results once in v1 before designing v2. The fixes came from
  v1 diagnostics (market correlation and turnover), and λ was tuned on train only, but a fully clean check would need
  a fresh period or universe.
- **The universe is small and liquid.** These are 104 mega-caps, where simple signals are heavily traded away. Real
  systematic funds use thousands of stocks and many more data sets.
- **Costs are a flat 5 bps.** Real costs depend on spread, size and volatility. There's no borrow cost for shorts.
- **Sharpe 0.38 is weak on its own.** Funds like Trexquant combine thousands of small, weakly correlated signals. One
  signal at this level is a building block, not a strategy.

## What I'd do next

1. Use a larger, point-in-time universe (historical index membership) to remove survivorship bias.
2. Add sector neutrality, and test more signal families (value, earnings surprise, volume-based).
3. Combine signals with a regularised regression fitted on rolling windows instead of a fixed IC weighting.
4. Model costs as a function of each stock's spread and volatility.
