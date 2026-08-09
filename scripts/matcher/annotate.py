"""Launch the one-time Hesban fracture/rim annotation pass."""
from __future__ import annotations
import argparse, threading, webbrowser
from pathlib import Path
from catalog.matcher_annotation import create_annotation_app

def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--expected-count", type=int, default=30)
    parser.add_argument("--set-name", default="hesban_30")
    parser.add_argument("--sorted-files", action="store_true")
    parser.add_argument("--invert-masks", action="store_true")
    parser.add_argument("--preview-before-save", action="store_true")
    parser.add_argument("--display-name", default="Hesban queries")
    args=parser.parse_args()
    app=create_annotation_app(args.project, args.queries,
                              expected_count=args.expected_count,
                              set_name=args.set_name,
                              sorted_files=args.sorted_files,
                              invert_masks=args.invert_masks,
                              preview_before_save=args.preview_before_save,
                              display_name=args.display_name)
    url=f"http://127.0.0.1:{args.port}"
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(
        f"Opening {url}. Press Ctrl+C after all {args.expected_count} are saved.",
        flush=True,
    )
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
    return 0
if __name__ == "__main__": raise SystemExit(main())
