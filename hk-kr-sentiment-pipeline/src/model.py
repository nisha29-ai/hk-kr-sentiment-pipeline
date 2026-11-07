from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

FEATURES = ["sentiment", "mention_count", "avg_engagement", "return_1d", "return_5d", "volume_change", "sma_5", "sma_20"]


def add_daily_sentiment(price: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
    out = price.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True).dt.normalize().astype("datetime64[ns, UTC]")
    if posts.empty:
        out["sentiment"] = 0.0; out["mention_count"] = 0; out["avg_engagement"] = 0.0
        return out
    p = posts.copy(); p["date"] = pd.to_datetime(p["timestamp"], utc=True).dt.normalize().astype("datetime64[ns, UTC]")
    weekend = p["date"].dt.day_of_week >= 5
    p.loc[weekend, "date"] = p.loc[weekend, "date"] - pd.offsets.BDay(1)
    p["engagement"] = p[["likes", "reposts", "views"]].fillna(0).sum(axis=1)
    daily = p.groupby(["ticker", "date"], as_index=False).agg(sentiment=("sentiment", "mean"), mention_count=("text", "count"), avg_engagement=("engagement", "mean"))
    aligned = []
    for ticker, prices_one in out.groupby("ticker"):
        prices_one = prices_one.sort_values("date").copy()
        posts_one = daily[daily["ticker"] == ticker].sort_values("date").copy()
        if posts_one.empty:
            prices_one["sentiment"] = 0.0; prices_one["mention_count"] = 0; prices_one["avg_engagement"] = 0.0
        else:
            prices_one = pd.merge_asof(prices_one, posts_one.drop(columns="ticker"), on="date", direction="backward")
            prices_one["sentiment"] = prices_one["sentiment"].fillna(0)
            prices_one["mention_count"] = prices_one["mention_count"].fillna(0)
            prices_one["avg_engagement"] = prices_one["avg_engagement"].fillna(0)
        aligned.append(prices_one)
    return pd.concat(aligned, ignore_index=True)


def fit_and_validate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    usable = frame.dropna(subset=FEATURES + ["target_next_return"]).copy()
    if len(usable) < 12:
        return pd.DataFrame(), {"error": "Not enough complete daily rows for validation."}
    X, y = usable[FEATURES], usable["target_next_return"]
    n_splits = min(3, max(2, len(usable) // 5))
    splitter = TimeSeriesSplit(n_splits=n_splits)
    predictions = []
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, l2_regularization=0.2, random_state=42)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        fold_metrics.append({"fold": fold, "mae": mean_absolute_error(y.iloc[test_idx], pred), "rmse": mean_squared_error(y.iloc[test_idx], pred) ** 0.5})
        temp = usable.iloc[test_idx][["date", "ticker", "close"]].copy(); temp["actual_return"] = y.iloc[test_idx].values; temp["predicted_return"] = pred; predictions.append(temp)
    final_model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, l2_regularization=0.2, random_state=42).fit(X, y)
    latest = frame.dropna(subset=FEATURES).sort_values("date").groupby("ticker").tail(1).copy()
    latest["predicted_return"] = final_model.predict(latest[FEATURES])
    latest["predicted_next_price"] = latest["close"] * (1 + latest["predicted_return"])
    latest["predicted_direction"] = np.where(latest["predicted_return"] >= 0, "up", "down")
    return pd.concat(predictions, ignore_index=True), {"folds": fold_metrics, "latest": latest.to_dict("records")}
