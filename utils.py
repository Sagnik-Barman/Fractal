import yfinance as yf
import numpy as np
import pandas as pd
from scipy import signal

def fetch_weekly_nifty(start="2007-09-17", end="2022-01-24", ticker="^NSEI"):
    """Download daily prices and resample to Friday-weekly Open/Close bars.

    Returns a DataFrame with columns ['Date', 'Open', 'Close', 't'], where 't'
    is position in the series rescaled to [0, 1] for the fractal interpolation.
    """
    df = yf.download(ticker, start=start, end=end, progress=False)

    # An empty frame here means the download failed (no network, rate limit, or
    # a delisted/renamed ticker). Fail loudly rather than a few lines later with
    # an opaque KeyError on the missing date column.
    if df is None or len(df) == 0:
        raise RuntimeError(
            f"No data returned for ticker {ticker!r} between {start} and {end}. "
            "Check network access to Yahoo Finance and that the ticker is valid."
        )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.columns = [str(col).split("_")[0] for col in df.columns]

    missing = {'Open', 'Close'} - set(df.columns)
    if missing:
        raise RuntimeError(f"downloaded frame is missing column(s): {sorted(missing)}")

    dfw = df.resample('W-FRI').agg({'Open': 'first', 'Close': 'last'})
    # Name the index explicitly so reset_index() always yields a 'Date' column,
    # whatever the provider happened to call it.
    dfw.index.name = 'Date'
    dfw = dfw.dropna().reset_index()

    if len(dfw) < 2:
        raise RuntimeError(f"only {len(dfw)} weekly bar(s) after resampling; need at least 2")

    dfw['t'] = np.arange(len(dfw)) / (len(dfw) - 1)
    return dfw[['Date', 'Open', 'Close', 't']]

def extract_extrema(open_series, order=1):
    arr = np.asarray(open_series)
    max_idx = signal.argrelextrema(arr, np.greater, order=order)[0]
    min_idx = signal.argrelextrema(arr, np.less, order=order)[0]
    idx = np.sort(np.concatenate(([0], max_idx, min_idx, [len(arr)-1])))
    idx = np.unique(idx)
    return idx