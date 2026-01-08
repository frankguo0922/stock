from __future__ import annotations

import argparse
import re
import sys
from urllib.request import urlopen

import yaml


TWSE_ISIN_URL = "https://isin.twse.com.tw/isin/e_C_public.jsp?strMode=2"


def fetch_html(url: str) -> str:
    try:
        with urlopen(url, timeout=20) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}")
            return resp.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch TWSE ISIN list: {exc}")


def parse_codes(html: str) -> list[str]:
    # The first column typically contains "2330 \u3000 TSMC"; capture leading 4 digits.
    matches = re.findall(r">\s*(\d{4})\s", html)
    codes = []
    seen = set()
    for code in matches:
        if code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def build_universe(codes: list[str]) -> dict:
    stock_ids = [f"{c}.TW" for c in codes]
    return {"universe": {"stock_ids": stock_ids}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Taiwan stock universe list from TWSE ISIN.")
    parser.add_argument("--out", default="configs/universe_tw.yaml", help="Output YAML path")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of stock IDs")
    args = parser.parse_args()

    html = fetch_html(TWSE_ISIN_URL)
    codes = parse_codes(html)
    if not codes:
        raise RuntimeError("No 4-digit stock codes found. Check source HTML format.")

    if args.limit and args.limit > 0:
        codes = codes[: args.limit]

    universe = build_universe(codes)
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(universe, f, allow_unicode=False, sort_keys=False)

    print(f"Wrote {len(codes)} stock IDs to {args.out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
