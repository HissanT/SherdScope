"""Command-line entry point for lossless U-Net profile dataset preparation."""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

from training_curator.server import create_curator_app
from training_curator.unet import predict_pending, train_unet
from training_curator.workspace import (
    DEFAULT_DATASET_NAME,
    DEFAULT_DPI,
    CuratorWorkspaceError,
    build_dpi_pilot,
    prepare_workspace,
    workspace_path,
)


def _project_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not (path / "project.json").is_file():
        raise argparse.ArgumentTypeError(
            "project must point to a SherdScope project folder containing project.json"
        )
    return path


def _progress(current: int, total: int, label: str) -> None:
    width = len(str(total))
    print(f"[{current:>{width}}/{total}] {label}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and curate a versioned U-Net profile dataset without "
            "modifying SherdScope cards or accepted masks."
        )
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    def common(command):
        command.add_argument(
            "--project",
            required=True,
            type=_project_path,
            help="SherdScope project folder",
        )
        command.add_argument(
            "--dataset-name",
            default=DEFAULT_DATASET_NAME,
            help=f"Versioned training folder name (default: {DEFAULT_DATASET_NAME})",
        )
        command.add_argument(
            "--dpi",
            type=int,
            default=DEFAULT_DPI,
            help=f"Direct PDF crop DPI (default: {DEFAULT_DPI})",
        )
        command.add_argument(
            "--force",
            action="store_true",
            help="Regenerate candidate files; approved masks remain decision-tracked",
        )

    pilot = subcommands.add_parser(
        "pilot", help="Render a small old-versus-lossless DPI comparison report"
    )
    common(pilot)
    pilot.add_argument("--count", type=int, default=20)
    pilot.add_argument("--no-browser", action="store_true")

    prepare = subcommands.add_parser(
        "prepare", help="Prepare all reviewed candidates and migrated draft masks"
    )
    common(prepare)
    prepare.add_argument("--limit", type=int)

    curate = subcommands.add_parser(
        "curate", help="Prepare/resume the dataset and start the local curator"
    )
    common(curate)
    curate.add_argument("--limit", type=int)
    curate.add_argument("--host", default="127.0.0.1")
    curate.add_argument("--port", type=int, default=8765)
    curate.add_argument("--no-browser", action="store_true")
    curate.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Start an already-prepared workspace immediately",
    )

    train = subcommands.add_parser(
        "train", help="Train a compact U-Net and build an unseen holdout preview"
    )
    train.add_argument("--project", required=True, type=_project_path)
    train.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    train.add_argument("--image-size", type=int, default=320)
    train.add_argument("--epochs", type=int, default=24)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--no-browser", action="store_true")

    predict = subcommands.add_parser(
        "predict", help="Generate separate U-Net masks for all pending candidates"
    )
    predict.add_argument("--project", required=True, type=_project_path)
    predict.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    predict.add_argument("--checkpoint", type=Path)
    predict.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "pilot":
            result = build_dpi_pilot(
                args.project,
                dataset_name=args.dataset_name,
                dpi=args.dpi,
                count=args.count,
                force=args.force,
                progress=_progress,
            )
            print(f"\nPilot complete: {result['report']}")
            if not args.no_browser:
                webbrowser.open(result["report"].as_uri())
            return 0

        if args.command == "prepare":
            result = prepare_workspace(
                args.project,
                dataset_name=args.dataset_name,
                dpi=args.dpi,
                limit=args.limit,
                force=args.force,
                progress=_progress,
            )
            print(
                "\nPreparation complete: "
                f"{result['rendered']} rendered, {result['reused']} reused, "
                f"{result['total_entries']} total candidates."
            )
            print(f"Workspace: {result['workspace']}")
            return 0

        if args.command == "train":
            root = workspace_path(args.project, args.dataset_name)
            if not (root / "manifest.json").is_file():
                raise CuratorWorkspaceError(
                    "The workspace has not been prepared; run the prepare command first"
                )
            result = train_unet(
                root,
                image_size=args.image_size,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                patience=args.patience,
                progress=lambda message: print(message, flush=True),
            )
            print(
                "\nTraining complete: "
                f"validation Dice={result['validation']['dice']:.4f}, "
                f"test Dice={result['test']['dice']:.4f}, "
                f"threshold={result['threshold']:.2f}"
            )
            print(f"Checkpoint: {result['checkpoint']}")
            print(f"Holdout preview: {result['preview']}")
            if not args.no_browser:
                webbrowser.open(result["preview"].as_uri())
            return 0

        if args.command == "predict":
            root = workspace_path(args.project, args.dataset_name)
            checkpoint = args.checkpoint or root / "models" / "unet_v1" / "best.pt"
            if not checkpoint.is_file():
                raise CuratorWorkspaceError(
                    f"Checkpoint does not exist: {checkpoint}. Run train first."
                )
            result = predict_pending(
                root, checkpoint, force=args.force, progress=_progress
            )
            print(
                "\nPrediction complete: "
                f"{result['generated']} generated, {result['reused']} reused, "
                f"{result['pending']} pending candidates."
            )
            print(f"Predictions: {result['output']}")
            print("Restart the curator to review the model masks.")
            return 0

        root = workspace_path(args.project, args.dataset_name)
        if not args.skip_prepare:
            result = prepare_workspace(
                args.project,
                dataset_name=args.dataset_name,
                dpi=args.dpi,
                limit=args.limit,
                force=args.force,
                progress=_progress,
            )
            root = result["workspace"]
            print(
                "\nPreparation complete: "
                f"{result['rendered']} rendered, {result['reused']} reused."
            )
        elif not (root / "manifest.json").is_file():
            raise CuratorWorkspaceError(
                "The workspace has not been prepared. Remove --skip-prepare."
            )

        if args.host not in {"127.0.0.1", "localhost"}:
            raise CuratorWorkspaceError(
                "The temporary curator is local-only; host must be 127.0.0.1 or localhost"
            )
        if not 1024 <= args.port <= 65535:
            raise CuratorWorkspaceError("Port must be between 1024 and 65535")
        url = f"http://127.0.0.1:{args.port}/"
        app = create_curator_app(root)
        print(f"\nCurator running at {url}")
        print("Press Ctrl+C in this terminal when you are finished.")
        if not args.no_browser:
            threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        app.run(
            host="127.0.0.1",
            port=args.port,
            debug=False,
            use_reloader=False,
            threaded=True,
        )
        return 0
    except CuratorWorkspaceError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nCurator stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
