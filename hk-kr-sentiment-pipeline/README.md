# HK/KR Sentiment Price Prediction Agent

Internship assessment implementation for a Python pipeline that combines social sentiment, recent news, market data, and next-period price prediction for five Hong Kong and five South Korean assets.

## What is included

- Apify dataset/actor adapter for social posts.
- Exa news adapter (optional; the pipeline runs without it in demo mode).
- Free Yahoo Finance price ingestion through `yfinance`.
- VADER sentiment baseline, engagement features, mentions, direction, support/resistance, and moving averages.
- Chronological `TimeSeriesSplit` validation to avoid random look-ahead leakage.
- HistGradientBoosting regression for next-day return and next price.
- CSV outputs, JSON metrics, charts, and a self-contained HTML report.
- Demo social data fallback so reviewers can run the project before adding API keys.

## Quick start

```bash
cd hk-kr-sentiment-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python run_pipeline.py
```

Open `outputs/report.html`. In demo mode, synthetic social posts are used only to verify that the pipeline executes; clearly label live versus demo results in your submission.

## Live API setup

Put keys in `.env`, never in source code or Git history. Set `APIFY_API_TOKEN` and either `APIFY_DATASET_ID` for an existing run or `APIFY_ACTOR_ID` plus `RUN_APIFY_ACTOR=true`. The selected actor's input schema may need to be adapted in `src/collectors.py`; record the final actor ID and input in your submission. Set `EXA_API_KEY` to enable recent-news collection. The current runner uses news collection as an adapter; extend `run_pipeline.py` to call `collect_exa_news()` per ticker and merge `news_sentiment` into the daily features.

## Data contract for Apify

The normalizer accepts common field names: `text`/`content`/`body`, `timestamp`/`createdAt`, `likes`, `reposts`, `views`, and `followers`. For another actor, add its exact field mapping in `src/sentiment.py`. Use two documented sources per exchange, for example one Apify actor for Naver Finance and one for LIHKG/Telegram, subject to the sources' terms and access rules. Do not scrape private communities or bypass access controls.

## Methodology

Social posts are timestamp-normalized to UTC and aggregated by ticker/day. VADER gives a transparent baseline compound score; for Korean-language posts, replace or supplement it with a Korean model and document the model. Market features include one- and five-day returns, volume change, 5/20-day moving averages, and 10-day support/resistance. The target is next-day return, converted to predicted next price. Validation is chronological and reports MAE and RMSE per fold.

## Suggested improvements before submission

1. Connect two real Apify sources per exchange and save raw exports under `data/raw/`.
2. Add explicit ticker dictionaries and keyword matching for each source.
3. Add a Korean sentiment model and English/HK slang handling; compare against VADER.
4. Add Exa news sentiment to the feature table and report source URLs.
5. Add a naïve previous-close baseline and directional accuracy.
6. Include a data dictionary, API limitations, ethics/terms-of-use note, and a screenshot of the HTML report.
7. Never email or commit keys in public; provide them privately only if the evaluator explicitly requests them.

## Project structure

```text
run_pipeline.py       entry point
src/collectors.py     Apify and Exa adapters
src/sentiment.py      post/news normalization and sentiment
src/market.py         price and technical features
src/model.py          features, model, time-series validation
src/report.py         charts and HTML report
outputs/              generated deliverables
```
