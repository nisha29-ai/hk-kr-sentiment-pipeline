from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

@dataclass(frozen=True)
class Settings:
    apify_token: str = os.getenv("APIFY_API_TOKEN", "")
    apify_dataset_id: str = os.getenv("APIFY_DATASET_ID", "")
    apify_actor_id: str = os.getenv("APIFY_ACTOR_ID", "")
    run_apify_actor: bool = os.getenv("RUN_APIFY_ACTOR", "false").lower() == "true"
    exa_key: str = os.getenv("EXA_API_KEY", "")
    hk_tickers: tuple[str, ...] = tuple(os.getenv("HK_TICKERS", "0700.HK,9988.HK,3690.HK,0005.HK,1299.HK").split(","))
    kr_tickers: tuple[str, ...] = tuple(os.getenv("KR_TICKERS", "005930.KS,000660.KS,035420.KS,035720.KS,005380.KS").split(","))
    lookback_days: int = 10
    telegram_export_path: str = os.getenv("TELEGRAM_EXPORT_PATH", str(ROOT / "data" / "raw" / "telegram_export.json"))

SETTINGS = Settings()
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
for folder in (RAW, PROCESSED, OUTPUTS):
    folder.mkdir(parents=True, exist_ok=True)
