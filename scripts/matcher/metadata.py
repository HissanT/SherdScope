"""Launch the one-time Hesban query metadata entry screen."""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from pathlib import Path

from catalog.matcher_metadata_curator import create_metadata_curator_app


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enter optional human metadata for the saved Hesban queries."
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--set-name", default="hesban_30")
    parser.add_argument("--expected-count", type=int, default=30)
    parser.add_argument(
        "--targets",
        type=Path,
        help="Optional JSON file containing a queries object with figure/item targets.",
    )
    args = parser.parse_args()
    real_targets = [
        ("4.5", "14"), ("3.53", "4"), ("3.84", "2"),
        ("3.88", "4"), ("3.25", "25"),
    ]
    targets = real_targets if args.set_name == "real_sherds_5" else None
    if args.targets:
        with args.targets.open(encoding="utf-8") as handle:
            target_rows = (json.load(handle).get("queries") or {})
        missing = [
            number
            for number in range(1, args.expected_count + 1)
            if str(number) not in target_rows
        ]
        if missing:
            parser.error(f"Target file is missing queries: {missing}")
        targets = [
            (
                str(target_rows[str(number)].get("figure") or ""),
                str(target_rows[str(number)].get("item") or ""),
            )
            for number in range(1, args.expected_count + 1)
        ]
        invalid = [
            number
            for number, (figure, item) in enumerate(targets, start=1)
            if not figure or not item
        ]
        if invalid:
            parser.error(f"Target file has blank figure/item values: {invalid}")
    app = create_metadata_curator_app(
        args.project, args.queries, set_name=args.set_name,
        expected_count=args.expected_count, targets=targets,
    )
    url = f"http://127.0.0.1:{args.port}"
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"Opening {url}. Press Ctrl+C when metadata entry is finished.", flush=True)
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
