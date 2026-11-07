from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.collectors import collect_apify_posts, collect_exa_news
from src.config import SETTINGS, OUTPUTS
from src.market import prices, technical_features
from src.model import add_daily_sentiment, fit_and_validate
from src.report import build_report
from src.sentiment import normalize_posts, normalize_news, normalize_xueqiu


def demo_posts(tickers: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(42); rows = []
    texts = ["strong earnings and growth outlook", "risk of slowdown, cautious", "fair value and positive momentum", "uncertain market, wait and see"]
    for ticker in tickers:
        for day in range(10):
            for i in range(3):
                rows.append({"ticker": ticker, "exchange": "HK" if ticker.endswith("HK") else "KR", "timestamp": datetime.now(timezone.utc)-timedelta(days=day, hours=i), "text": texts[(day+i)%len(texts)], "likes": int(rng.integers(1,100)), "reposts": int(rng.integers(0,30)), "views": int(rng.integers(20,1000)), "followers": int(rng.integers(10,5000))})
    frame = pd.DataFrame(rows)
    from src.sentiment import score_text
    frame["sentiment"] = frame["text"].map(score_text); frame["direction"] = frame["sentiment"].map(lambda x: "bullish" if x >= .05 else "bearish" if x <= -.05 else "neutral")
    return frame


def main() -> None:
    tickers = list(SETTINGS.hk_tickers + SETTINGS.kr_tickers)
    raw = collect_apify_posts()
    export_path = Path(SETTINGS.telegram_export_path)
    if raw:
        posts = normalize_posts(raw, "UNKNOWN", "UNKNOWN")
        source_mode = "live Apify API"
    elif export_path.exists():
        telegram_items = json.loads(export_path.read_text(encoding="utf-8"))
        posts = normalize_posts(telegram_items, "UNKNOWN", "KR")
        posts["source"] = "telegram"
        source_mode = "real Telegram export"
    else:
        posts = demo_posts(tickers)
        source_mode = "demo social data"
    xueqiu_path = ROOT / "data" / "raw" / "xueqiu_export.json"
    if xueqiu_path.exists() and not raw:
        xueqiu_items = json.loads(xueqiu_path.read_text(encoding="utf-8"))
        ticker_path = ROOT / "data" / "raw" / "xueqiu_ticker_export.json"
        if ticker_path.exists():
            xueqiu_items += json.loads(ticker_path.read_text(encoding="utf-8"))
        xueqiu_posts = normalize_xueqiu(xueqiu_items).drop_duplicates(subset=["ticker", "timestamp", "text"])
        if not xueqiu_posts.empty:
            posts = pd.concat([posts, xueqiu_posts], ignore_index=True, sort=False)
            source_mode += " + real Xueqiu export"
    daum_path = ROOT / "data" / "raw" / "daum_finance_export.json"
    if daum_path.exists() and not raw:
        daum_items = json.loads(daum_path.read_text(encoding="utf-8"))
        daum_posts = normalize_posts(daum_items, "UNKNOWN", "KR")
        if not daum_posts.empty:
            posts = pd.concat([posts, daum_posts], ignore_index=True, sort=False)
            source_mode += " + real Daum Finance export"
    if raw:
        # Map a scraped item to a ticker by matching ticker/cashtag text.
        posts["ticker"] = posts.apply(lambda r: next((t for t in tickers if t.split('.')[0] in r.text.upper()), tickers[0]), axis=1)
        posts["exchange"] = posts["ticker"].map(lambda t: "HK" if t.endswith("HK") else "KR")
    prices_frame = prices(tickers)
    if prices_frame.empty:
        raise RuntimeError("No market data returned. Check internet access or tickers.")
    price_features = technical_features(prices_frame)
    merged = add_daily_sentiment(price_features, posts)
    predictions, metrics = fit_and_validate(merged)
    latest = pd.DataFrame(metrics.get("latest", []))
    if not latest.empty:
        latest["exchange"] = latest["ticker"].map(lambda t: "HK" if t.endswith("HK") else "KR")
    report = build_report(latest, posts, metrics, OUTPUTS)
    merged.to_csv(OUTPUTS / "feature_table.csv", index=False)
    predictions.to_csv(OUTPUTS / "validation_predictions.csv", index=False)
    news_frames = []
    if SETTINGS.exa_key:
        for ticker in tickers:
            news_frames.append(normalize_news(collect_exa_news(ticker), ticker))
    news = pd.concat(news_frames, ignore_index=True) if news_frames else pd.DataFrame()
    news.to_csv(OUTPUTS / "exa_news.csv", index=False)
    (OUTPUTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(f"Report: {report}")
    print(f"Mode: {source_mode}; posts={len(posts)}; price rows={len(prices_frame)}; exa_news={len(news)}")

if __name__ == "__main__":
    main()
