from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ANALYZER = SentimentIntensityAnalyzer()
POSITIVE_TERMS = {
    "매수", "유리", "희소식", "대박", "호황", "수주", "폭등", "상승", "성장", "기대", "증가", "호재", "폭발", "강세", "buy", "bullish", "growth", "strong", "profit", "upgrade",
    "利好", "大涨", "增长", "强劲", "扩容", "前列", "好消息", "买入", "上涨", "机会", "增长", "盈利", "看好"
}
NEGATIVE_TERMS = {
    "폭락", "악재", "감소", "하락", "불가피", "약세", "위험", "손실", "빼앗김", "급감", "downside", "bearish", "risk", "loss", "weak", "decline",
    "暴跌", "下跌", "风险", "亏损", "下降", "恶化", "利空", "担忧", "熊市", "回调"
}


def score_text(text: str) -> float:
    text = str(text or "").strip()
    if not text:
        return 0.0
    vader = float(ANALYZER.polarity_scores(text)["compound"])
    lowered = text.lower()
    positive = sum(lowered.count(term.lower()) for term in POSITIVE_TERMS)
    negative = sum(lowered.count(term.lower()) for term in NEGATIVE_TERMS)
    if positive + negative == 0:
        return vader
    lexicon_score = (positive - negative) / (positive + negative)
    return float(0.7 * lexicon_score + 0.3 * vader)


def _first(item: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in item and item[key] not in (None, ""):
            return item[key]
    return default


def normalize_posts(items: list[dict[str, Any]], ticker: str, exchange: str) -> pd.DataFrame:
    rows = []
    for item in items:
        text = _first(item, "text", "content", "body", "title", default="")
        timestamp = _first(item, "timestamp", "datetime", "createdAt", "created_at", "publishedAt", default=datetime.now(timezone.utc).isoformat())
        try:
            timestamp = pd.to_datetime(timestamp, utc=True)
        except Exception:
            timestamp = pd.Timestamp.now(tz="UTC")
        rows.append({
            "ticker": _first(item, "ticker", default=ticker), "exchange": _first(item, "exchange", default=exchange), "timestamp": timestamp,
            "text": str(text), "likes": float(_first(item, "likes", "likeCount", "like_count", default=0) or 0),
            "reposts": float(_first(item, "reposts", "retweets", "shareCount", default=0) or 0),
            "views": float(_first(item, "views", "viewCountParsed", "viewCount", default=0) or 0),
            "followers": float(_first(item, "followers", "authorFollowers", default=0) or 0),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["ticker", "exchange", "timestamp", "text", "likes", "reposts", "views", "followers", "sentiment"])
    frame["sentiment"] = frame["text"].map(score_text)
    frame["direction"] = frame["sentiment"].map(lambda x: "bullish" if x >= 0.05 else "bearish" if x <= -0.05 else "neutral")
    return frame


def normalize_news(items: list[dict[str, Any]], ticker: str) -> pd.DataFrame:
    rows = []
    for item in items:
        text = " ".join([str(_first(item, "title", default="")), str(_first(item, "text", default=""))])
        rows.append({"ticker": ticker, "published": _first(item, "publishedDate", "published", default=""), "title": _first(item, "title", default=""), "url": _first(item, "url", default=""), "news_sentiment": score_text(text)})
    return pd.DataFrame(rows)


def normalize_xueqiu(items: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize Xueqiu exports and retain only explicit HK tickers."""
    rows = []
    for item in items:
        tickers = item.get("tickersInPost") or []
        if not tickers and item.get("tickerExtracted"):
            tickers = [item["tickerExtracted"]]
        hk_tickers = [t for t in tickers if str(t).upper().startswith("HK") and str(t)[2:].isdigit()]
        for raw_ticker in hk_tickers:
            code = str(int(str(raw_ticker)[2:])).zfill(4)
            metrics = item.get("metrics") or {}
            rows.append({
                "ticker": f"{code}.HK", "exchange": "HK",
                "timestamp": pd.to_datetime(item.get("publishedAt"), utc=True),
                "text": str(item.get("postText") or ""),
                "likes": float(metrics.get("likeCount") or 0),
                "reposts": float(metrics.get("retweetCount") or 0),
                "views": float(metrics.get("viewCount") or 0),
                "followers": float((item.get("author") or {}).get("followerCount") or 0),
                "source": "xueqiu", "url": item.get("postUrl", ""),
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["sentiment"] = frame["text"].map(score_text)
    frame["direction"] = frame["sentiment"].map(lambda x: "bullish" if x >= 0.05 else "bearish" if x <= -0.05 else "neutral")
    return frame
