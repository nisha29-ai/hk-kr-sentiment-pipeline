from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from .config import SETTINGS, RAW


def _save(name: str, data: Any) -> None:
    (RAW / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def collect_apify_posts() -> list[dict[str, Any]]:
    """Read an Apify dataset, or start an actor when configured.

    Actor input is intentionally generic: replace ACTOR_INPUT below with the
    input schema of the selected Apify actor and document the actor ID in README.
    """
    if not SETTINGS.apify_token:
        return []
    headers = {"Authorization": f"Bearer {SETTINGS.apify_token}"}
    dataset_id = SETTINGS.apify_dataset_id
    if SETTINGS.run_apify_actor and SETTINGS.apify_actor_id:
        url = f"https://api.apify.com/v2/acts/{SETTINGS.apify_actor_id}/runs?waitForFinish=120"
        actor_input = {"maxItems": 500, "proxy": {"useApifyProxy": True}}
        response = requests.post(url, headers=headers, json={"input": actor_input}, timeout=150)
        response.raise_for_status()
        dataset_id = response.json()["data"]["defaultDatasetId"]
    if not dataset_id:
        return []
    url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?clean=true&format=json"
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    posts = response.json()
    _save("apify_posts.json", posts)
    return posts if isinstance(posts, list) else []


def collect_exa_news(query: str, start_date: datetime | None = None) -> list[dict[str, Any]]:
    if not SETTINGS.exa_key:
        return []
    start_date = start_date or datetime.now(timezone.utc) - timedelta(days=SETTINGS.lookback_days)
    payload = {
        "query": f"{query} stock market financial news",
        "startPublishedDate": start_date.isoformat().replace("+00:00", "Z"),
        "numResults": 10,
        "contents": {"text": {"maxCharacters": 3000}},
    }
    response = requests.post(
        "https://api.exa.ai/search",
        headers={"x-api-key": SETTINGS.exa_key, "Authorization": f"Bearer {SETTINGS.exa_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    _save("exa_news.json", results)
    return results
