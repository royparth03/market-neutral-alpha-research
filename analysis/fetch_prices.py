"""Download daily adjusted prices for ~100 large US stocks (2010-2025) from Yahoo's public chart API.

Run:  python analysis/fetch_prices.py   ->  data/prices.csv (adjusted close), data/volume.csv
Universe note: these are companies that are large today, so the backtest has survivorship bias
(stocks that failed or were delisted since 2010 are missing). This is stated in the README.
"""
import json
import os
import time
import urllib.request

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

TICKERS = """AAPL MSFT AMZN GOOGL NVDA JPM BAC WFC C GS MS JNJ PFE MRK ABT LLY UNH AMGN GILD BMY MDT TMO
DHR XOM CVX COP SLB OXY KO PEP PG CL KMB WMT COST HD LOW TGT MCD SBUX NKE DIS CMCSA VZ T INTC CSCO
ORCL IBM QCOM TXN ADBE CRM ACN INTU AMAT MU HON GE CAT DE MMM BA LMT RTX UPS FDX UNP CSX NSC EMR ETN
ITW AXP V MA BLK SCHW USB PNC COF MET PRU AIG ALL TRV CB SPG AMT PLD NEE DUK SO D AEP EXC SRE MO PM
HSY CVS CI ISRG BKNG""".split()

START, END = 1262304000, 1767225600  # 2010-01-01 to 2026-01-01 (UTC seconds)
URL = ("https://query1.finance.yahoo.com/v8/finance/chart/{t}?period1={a}&period2={b}"
       "&interval=1d&events=div,splits")


def fetch(ticker):
    req = urllib.request.Request(URL.format(t=ticker, a=START, b=END),
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.load(r)["chart"]["result"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s").normalize()
    adj = res["indicators"]["adjclose"][0]["adjclose"]
    vol = res["indicators"]["quote"][0]["volume"]
    return pd.Series(adj, index=idx, name=ticker), pd.Series(vol, index=idx, name=ticker)


def main():
    px, vol, failed = {}, {}, []
    for t in TICKERS:
        try:
            p, v = fetch(t)
            px[t], vol[t] = p, v
        except Exception as e:  # noqa: BLE001
            failed.append((t, str(e)[:60]))
        time.sleep(0.3)
    prices = pd.DataFrame(px).sort_index()
    prices = prices[~prices.index.duplicated()]
    volume = pd.DataFrame(vol).sort_index()
    volume = volume[~volume.index.duplicated()]
    prices.to_csv(os.path.join(DATA, "prices.csv"))
    volume.to_csv(os.path.join(DATA, "volume.csv"))
    print(f"saved {prices.shape[1]} tickers x {prices.shape[0]} days "
          f"({prices.index.min().date()} to {prices.index.max().date()})")
    if failed:
        print("failed:", failed)


if __name__ == "__main__":
    main()
