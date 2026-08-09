"""Launch the photo-assisted 68-real-sherd outline editor."""

from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

from catalog.real_sherd_annotation import create_real_sherd_annotation_app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--expected-count", type=int, default=68)
    parser.add_argument("--set-name", default="real_sherds_68")
    args = parser.parse_args()
    app = create_real_sherd_annotation_app(
        args.project,
        args.masks,
        args.photos,
        expected_count=args.expected_count,
        set_name=args.set_name,
    )
    url = f"http://127.0.0.1:{args.port}"
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"Opening {url}. Press Ctrl+C after all {args.expected_count} are saved.", flush=True)
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
