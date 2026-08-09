"""Generate deterministic PNG and SVG figures for the real-sherd evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from catalog.real_sherd_evaluation import load_evaluation, runtime, score_summary


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "outputs" / "real_sherd_68_2626_pool400_20260808"
COLOR = "#1f77b4"


def _save(figure, root: Path, name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(root / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _hist(values, *, title: str, xlabel: str, bins, output: Path, name: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(values, bins=bins, color=COLOR, edgecolor="white")
    axis.set(title=title, xlabel=xlabel, ylabel="Queries")
    axis.grid(axis="y", alpha=0.25)
    _save(figure, output, name)


def plot(run_dir: Path, output: Path) -> list[str]:
    _manifest, records = load_evaluation(run_dir)
    costs, margins, ranks, runtimes = [], [], [], []
    for item in records:
        run = item["record"]["run"]
        winner = run["results"][0]
        costs.append(float(winner["overall_score"]))
        if run.get("confidence_margin") is not None:
            margins.append(float(run["confidence_margin"]))
        rank = (winner.get("retrieval") or {}).get("rank")
        if rank is not None:
            ranks.append(int(rank))
        runtimes.append(runtime(run))

    made = []
    _hist(costs, title="Top-1 match-cost distribution", xlabel="Top-1 match cost (lower is better)", bins=12, output=output, name="top1_match_cost_distribution")
    made.append("top1_match_cost_distribution")
    _hist(margins, title="First–second margin distribution", xlabel="Rank-2 cost minus rank-1 cost", bins=12, output=output, name="first_second_margin")
    made.append("first_second_margin")
    maximum_rank = max(ranks or [1])
    rank_bins = np.linspace(0.5, maximum_rank + 0.5, min(21, maximum_rank + 1))
    _hist(ranks, title="Top-1 retrieval-rank distribution", xlabel="Retrieval rank in the 400-candidate pool", bins=rank_bins, output=output, name="retrieval_rank_histogram")
    made.append("retrieval_rank_histogram")

    figure, axis = plt.subplots(figsize=(13, 6))
    x = np.arange(1, len(runtimes) + 1)
    bottom = np.zeros(len(runtimes))
    palette = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    labels = [("retrieval", "Retrieval"), ("coarse", "Coarse"), ("medium", "Medium"), ("fine", "Fine")]
    for (key, label), color in zip(labels, palette):
        values = np.array([row[key] for row in runtimes])
        axis.bar(x, values, bottom=bottom, width=0.85, label=label, color=color)
        bottom += values
    axis.set(title="Matcher runtime by query", xlabel="Query", ylabel="Seconds", xlim=(0, len(runtimes) + 1))
    axis.legend(ncol=4)
    axis.grid(axis="y", alpha=0.2)
    _save(figure, output, "runtime_stacked_bar")
    made.append("runtime_stacked_bar")

    expert = score_summary(run_dir / "expert_scores.json")
    if expert and expert["scores"]:
        figure, axis = plt.subplots(figsize=(7, 5))
        counts = [expert["counts"][value] for value in range(4)]
        axis.bar(range(4), counts, color=["#d73027", "#fc8d59", "#91cf60", "#1a9850"])
        axis.set(title="Expert top-1 score distribution", xlabel="Expert score", ylabel="Queries", xticks=range(4))
        axis.grid(axis="y", alpha=0.25)
        _save(figure, output, "top1_expert_score_histogram")
        made.append("top1_expert_score_histogram")

        scores_value = read_score_by_query(run_dir / "expert_scores.json")
        paired = [(costs[number - 1], score) for number, score in scores_value.items() if 1 <= number <= len(costs)]
        if paired:
            px, py = np.array([v[0] for v in paired]), np.array([v[1] for v in paired])
            correlation = float(np.corrcoef(px, py)[0, 1]) if len(px) >= 2 and np.std(px) > 0 and np.std(py) > 0 else float("nan")
            figure, axis = plt.subplots(figsize=(7, 5))
            axis.scatter(px, py, color=COLOR, alpha=0.8)
            label = "undefined" if np.isnan(correlation) else f"r = {correlation:.3f}"
            axis.set(title=f"Top-1 cost vs expert score ({label})", xlabel="Top-1 match cost", ylabel="Expert score", yticks=range(4))
            axis.grid(alpha=0.25)
            _save(figure, output, "cost_vs_score_scatter")
            made.append("cost_vs_score_scatter")
    else:
        print("Expert-dependent plots skipped: expert_scores.json is absent or top-1 is unscored.")
    return made


def read_score_by_query(path: Path) -> dict[int, int]:
    from catalog.real_sherd_evaluation import read_json

    output = {}
    for number, query in (read_json(path).get("queries") or {}).items():
        top = next((row for row in query.get("candidates") or [] if int(row.get("rank") or 0) == 1), None)
        if top is not None and top.get("score") in {0, 1, 2, 3}:
            output[int(number)] = int(top["score"])
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run_dir / "figures"
    names = plot(args.run_dir, output)
    print(f"{output.resolve()} ({len(names)} figure sets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
