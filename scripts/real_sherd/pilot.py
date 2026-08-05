"""Prepare and review five real sherd photographs against the pottery U-Net."""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

from real_sherd_pilot.pipeline import DEFAULT_INPUT_SIZE, DEFAULT_THRESHOLD, PilotError, prepare_workspace
from real_sherd_pilot.server import create_app


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT.parent / "DL ArchProject" / "training" / "unet_pottery_final.pth"
DEFAULT_OUTPUT = ROOT / "outputs" / "real_sherd_pilot_5"
DEFAULT_IMAGES = [Path.home() / "Downloads" / f"IMG_{number}.JPG" for number in range(6450, 6455)]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Isolated five-photo real-sherd segmentation pilot")
    value.add_argument("--image", action="append", type=Path,
                       help="Photograph path; repeat exactly five times (defaults to IMG_6450-6454.JPG)")
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    value.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    value.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    value.add_argument("--force", action="store_true", help="Re-run initial predictions; originals stay untouched")
    value.add_argument("--prepare-only", action="store_true",
                       help="Prepare predictions and exit without starting the review server")
    value.add_argument("--host", default="127.0.0.1")
    value.add_argument("--port", type=int, default=8775)
    value.add_argument("--no-browser", action="store_true")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.host not in {"127.0.0.1", "localhost"}:
            raise PilotError("This review tool is local-only; host must be 127.0.0.1 or localhost")
        if not 1024 <= args.port <= 65535:
            raise PilotError("Port must be between 1024 and 65535")
        images = args.image or DEFAULT_IMAGES
        result = prepare_workspace(
            images, args.output, args.checkpoint, threshold=args.threshold,
            input_size=args.input_size, force=args.force,
            progress=lambda current, total, label: print(f"[{current}/{total}] U-Net: {label}", flush=True),
        )
        url = f"http://127.0.0.1:{args.port}/"
        print(f"\nPrepared {result['count']} photographs on {result['device']}.")
        print(f"Pilot data: {result['workspace']}")
        if args.prepare_only:
            print("Preparation-only run complete. Start again without --prepare-only to review.")
            return 0
        print(f"Review UI: {url}")
        print("Press Ctrl+C when finished. Progress is saved after every completed mask.")
        if not args.no_browser:
            threading.Timer(.8, lambda: webbrowser.open(url)).start()
        create_app(result["workspace"]).run(host="127.0.0.1", port=args.port,
                                              debug=False, use_reloader=False, threaded=True)
        return 0
    except PilotError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nPilot stopped; saved work is preserved.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
