"""Launch the one-time Hesban query metadata entry screen."""

from __future__ import annotations

import argparse
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
    args = parser.parse_args()
    real_targets = [
        ("4.5", "14"), ("3.53", "4"), ("3.84", "2"),
        ("3.88", "4"), ("3.25", "25"),
    ]
    targets = real_targets if args.set_name == "real_sherds_5" else None
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
