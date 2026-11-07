from __future__ import annotations

import pandas as pd
import yfinance as yf


def prices(tickers: list[str], period: str = "3mo") -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        data = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False, group_by="column")
        if data.empty:
            continue
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.reset_index()
        data.columns = [str(c).lower().replace(" ", "_") for c in data.columns]
        data["ticker"] = ticker
        frames.append(data)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def technical_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.sort_values(["ticker", "date"]).copy()
    grouped = frame.groupby("ticker", group_keys=False)
    frame["return_1d"] = grouped["close"].pct_change()
    frame["return_5d"] = grouped["close"].pct_change(5)
    frame["volume_change"] = grouped["volume"].pct_change()
    frame["sma_5"] = grouped["close"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    frame["sma_20"] = grouped["close"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    frame["support_10d"] = grouped["low"].transform(lambda x: x.rolling(10, min_periods=1).min())
    frame["resistance_10d"] = grouped["high"].transform(lambda x: x.rolling(10, min_periods=1).max())
    frame["target_next_return"] = grouped["close"].shift(-1) / frame["close"] - 1
    return frame
