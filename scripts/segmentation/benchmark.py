"""Run the five-photo segmentation comparison without changing saved gold masks."""

from __future__ import annotations

import argparse
from pathlib import Path

from real_sherd_pilot.benchmark import run_benchmark


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT / "outputs" / "real_sherd_pilot_5")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "real_sherd_segmentation_benchmark")
    parser.add_argument("--sam-checkpoint", type=Path, default=ROOT / "models" / "sam2.1_t.pt")
    parser.add_argument("--deeplab-checkpoint", type=Path, default=ROOT / "models" / "deeplabv3plus_pottery_best.pth")
    args = parser.parse_args()
    sam = args.sam_checkpoint if args.sam_checkpoint.is_file() else None
    deeplab = args.deeplab_checkpoint if args.deeplab_checkpoint.is_file() else None
    result = run_benchmark(args.workspace, args.output, sam_checkpoint=sam, deeplab_checkpoint=deeplab)
    print(f"Benchmark written to: {args.output.resolve()}")
    for method, metrics in result["aggregate"].items():
        print(f"{method:24s} Dice={metrics['dice']['mean']:.3f}  "
              f"IoU={metrics['iou']['mean']:.3f}  boundary-F1={metrics['boundary_f1_at_3px']['mean']:.3f}")


if __name__ == "__main__":
    main()
