#!/usr/bin/env python3
"""Trigger a MODIS LST ingest on your running OCS instance (the same call the /manage form makes).

Examples:
    python scripts/ingest_lst.py --dataset modis_lst_day_monthly   --start 2015-01 --end 2025-12
    python scripts/ingest_lst.py --dataset modis_lst_night_monthly --start 2015-01 --end 2025-12
    python scripts/ingest_lst.py --dataset modis_lst_day_daily     --start 2025-01-01 --end 2025-01-31
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import httpx

DATASETS = ("modis_lst_day_monthly", "modis_lst_night_monthly", "modis_lst_day_daily", "modis_lst_night_daily")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--start", required=True, help="YYYY-MM (monthly) or YYYY-MM-DD (daily)")
    p.add_argument("--end", required=True, help="a later end is fine: OCS stops at the newest data that exists")
    p.add_argument("--url", default=os.environ.get("OCS_URL", "http://127.0.0.1:9000"))
    p.add_argument("--overwrite", action="store_true", help="re-ingest periods already in the store")
    p.add_argument("--no-publish", action="store_true", help="ingest without publishing to STAC / map viewer")
    a = p.parse_args()

    payload = {"dataset_id": a.dataset, "start": a.start, "end": a.end,
               "overwrite": a.overwrite, "publish": not a.no_publish}
    r = httpx.post(f"{a.url}/ingestions", json=payload, timeout=None)  # streams period by period: can take a while
    print(json.dumps(r.json(), indent=2))
    return 0 if r.is_success else 1


if __name__ == "__main__":
    sys.exit(main())
