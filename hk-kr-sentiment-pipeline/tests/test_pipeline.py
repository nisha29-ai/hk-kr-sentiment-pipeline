import pandas as pd
from src.sentiment import normalize_posts, score_text
from src.model import add_daily_sentiment


def test_sentiment_direction():
    assert score_text("great growth and strong earnings") > 0
    assert score_text("terrible loss and risk") < 0


def test_post_normalization():
    frame = normalize_posts([{"text": "strong growth", "timestamp": "2026-01-01T00:00:00Z", "likes": 4}], "0700.HK", "HK")
    assert len(frame) == 1
    assert frame.iloc[0]["ticker"] == "0700.HK"


def test_daily_merge():
    prices = pd.DataFrame({"ticker": ["0700.HK"], "date": [pd.Timestamp("2026-01-01", tz="UTC")], "close": [100]})
    posts = normalize_posts([{"text": "bullish", "timestamp": "2026-01-01T01:00:00Z"}], "0700.HK", "HK")
    out = add_daily_sentiment(prices, posts)
    assert out.iloc[0]["mention_count"] == 1
