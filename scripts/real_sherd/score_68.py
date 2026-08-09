"""Launch the localhost-only expert scoring UI for 68 real-sherd runs."""

from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

from catalog.real_sherd_scoring import create_scoring_app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()
    app = create_scoring_app(
        args.project,
        args.run_dir,
        args.output,
        annotator=args.annotator,
        top_n=args.top_n,
    )
    url = f"http://127.0.0.1:{args.port}/"
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
