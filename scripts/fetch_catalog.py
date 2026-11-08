"""Save the published ``judgments`` catalog for linking runs off the build machine.

The local database holds only a few judgments, but linking needs every one.
This pages through data.zeeker.sg's JSON API (CSV export and custom SQL are
both disabled there) and keeps only the columns the linker uses.

    uv run python scripts/fetch_catalog.py [--out .cache/catalog.json]
"""

import argparse
import json
import time
from pathlib import Path

import httpx

URL = "https://data.zeeker.sg/zeeker-judgements/judgments.json"
COLUMNS = ["id", "citation", "case_name", "decision_date", "court", "source_url"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=".cache/catalog.json")
    args = parser.parse_args()

    # Sorting by rowid keeps each page fast; larger pages time out.
    params = [("_shape", "objects"), ("_size", "250"), ("_sort", "rowid")]
    params += [("_col", c) for c in COLUMNS]
    rows, after, failures = {}, None, 0
    with httpx.Client(
        timeout=90,
        follow_redirects=True,
        headers={"User-Agent": "ZeekerBot/1.0 (+https://data.zeeker.sg)"},
    ) as client:
        while True:
            r = client.get(URL, params=params + ([("_next", after)] if after else []))
            if r.status_code != 200:
                failures += 1
                if failures > 4:
                    r.raise_for_status()
                time.sleep(3)
                continue
            data = r.json()
            for row in data["rows"]:
                row.pop("rowid", None)
                rows[row["id"]] = row
            after = data.get("next")
            if not after:
                break
            time.sleep(0.2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(list(rows.values()), ensure_ascii=False))
    print(f"{len(rows)} judgments → {out}")


if __name__ == "__main__":
    main()
