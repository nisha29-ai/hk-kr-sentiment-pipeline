from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template

TEMPLATE = """<!doctype html><html><head><meta charset='utf-8'><title>HK/KR Sentiment Report</title><style>body{font-family:Arial;max-width:1200px;margin:30px auto;color:#17202a}table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border:1px solid #ddd;padding:7px;text-align:right}th{background:#eef3f8}td:first-child,th:first-child{text-align:left}.card{display:inline-block;padding:12px;margin:5px;background:#f5f7fa;border-radius:8px}</style></head><body><h1>HK/KR Sentiment Price Prediction</h1><p>Generated {{ generated }}. Price source: Yahoo Finance. Sentiment: VADER plus Korean/Chinese financial lexicon. Predictions use chronological validation.</p><div class='card'>Assets: {{ assets }}</div><div class='card'>Posts: {{ posts }}</div><div class='card'>Validation folds: {{ folds }}</div><h2>Latest ranked predictions</h2>{{ table|safe }}<h2>Charts</h2><img src='mention_sentiment.png' width='900'><br><img src='prediction.png' width='900'></body></html>"""


def build_report(latest: pd.DataFrame, posts: pd.DataFrame, metrics: dict, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    if latest.empty:
        latest = pd.DataFrame(columns=["ticker", "exchange", "close", "sentiment", "mention_count", "predicted_next_price", "predicted_direction"])
    latest = latest.sort_values(["exchange", "mention_count", "sentiment"], ascending=[True, False, False])
    plt.figure(figsize=(11, 6)); sns.barplot(data=latest, x="ticker", y="mention_count", hue="predicted_direction", dodge=False); plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(output / "mention_sentiment.png", dpi=140); plt.close()
    if not latest.empty:
        plt.figure(figsize=(11, 6)); sns.scatterplot(data=latest, x="close", y="predicted_next_price", hue="exchange", s=140); plt.axline((0,0), slope=1, color="gray", linestyle="--"); plt.tight_layout(); plt.savefig(output / "prediction.png", dpi=140); plt.close()
    html = Template(TEMPLATE).render(generated=pd.Timestamp.now(tz="UTC"), assets=len(latest), posts=len(posts), folds=len(metrics.get("folds", [])), table=latest.to_html(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
    (output / "report.html").write_text(html, encoding="utf-8")
    return output / "report.html"
