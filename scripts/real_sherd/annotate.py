"""Open the fracture/rim-point editor for the five real-sherd queries."""
from __future__ import annotations

import argparse
import shutil
import threading
import webbrowser
from pathlib import Path

from catalog.matcher_annotation import create_annotation_app

PILOT_IMAGE_BY_QUERY = {number: 6449 + number for number in range(1, 6)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate fracture and gold/rim points for real-sherd queries")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--pilot-output", type=Path, default=Path("outputs/real_sherd_pilot_5"))
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()

    pilot_masks = args.pilot_output.resolve() / "exports" / "manual_gold_queries"
    # Keep this queue separate from older QueryN.png files.  Having both
    # Query2.png and query_2.png made the dictionary loader occasionally pick
    # the stale uppercase file for the same query number.
    temp_queries = args.pilot_output.resolve() / "annotation_queries_current"
    temp_queries.mkdir(parents=True, exist_ok=True)
    for number in range(1, 6):
        source = pilot_masks / f"IMG_{PILOT_IMAGE_BY_QUERY[number]}_manual.png"
        if not source.is_file():
            raise SystemExit(f"Missing manual mask for Query {number}: {source}")
        shutil.copy2(source, temp_queries / f"Query{number}.png")

    app = create_annotation_app(
        args.project.resolve(), temp_queries, expected_count=5, set_name="real_sherds_5"
    )
    url = f"http://127.0.0.1:{args.port}"
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"Opening {url}. Annotate Query 2, then press Enter to save.", flush=True)
    print("F = fracture trace; G = gold/rim point; Clear = remove old points.", flush=True)
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
