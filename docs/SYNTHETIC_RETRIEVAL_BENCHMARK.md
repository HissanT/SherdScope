# Synthetic retrieval benchmark

This benchmark is read-only with respect to reviewed masks, contour artifacts,
saved query runs, and matcher results. It generates queries in memory and writes
only to the explicitly selected output directory.

## Inputs

The command requires a SherdScope project directory that can be read by
`catalog.contours.load_ready_artifacts`. Every eligible parent must have:

- a unique `reference_id`;
- an approved `reference_master_boundary`;
- `source_points` in original upright publication coordinates;
- a valid `nominal_seam_fraction`.

The scaling comparison accepts an earlier `synthetic_benchmark.json`. A schema-1
report from the 1,058-reference run is supported for combined retrieval metrics.
New schema-2 reports also contain Wilson intervals and outline/ribbon results.

## Default 300-parent design

The default samples 300 parents without replacement using a fixed seed. It
creates seven queries per parent, or 2,100 queries total:

- `grid_clean`, `clean`, `light`, and `moderate`;
- `partial_75`, `partial_50`, and `partial_25`.

The three dose conditions retain exactly 75%, 50%, or 25% of total profile arc
length, symmetrically around the validated rim. They share the same seeded
rotation, scale, and translation for each parent. The clean/light/moderate
conditions remain paired in the same way, with only their noise dose changing.

## Safe command for an existing 2,600-reference index

Use a new output directory for every run. Do not pass `--full-rerank-limit`;
its default is zero, so the run remains retrieval-only.

```powershell
python -m scripts.matcher.benchmark <PROJECT_PATH> `
  --output outputs/synthetic_300_parent_2600_index_20260807 `
  --sample-size 300 `
  --require-existing-index `
  --comparison baseline_1058=outputs/20260801_synthetic_corrected_rim/synthetic_benchmark.json
```

`--require-existing-index` is the concurrency-safe option: it fails before the
benchmark if the shared retrieval cache is missing or stale instead of building
or replacing that cache while another matcher may be running. The benchmark
still consumes CPU and memory, so wait for other long matcher jobs to finish
before launching the 2,100-query run.

The command refuses to replace existing benchmark artifacts unless
`--overwrite-output` is explicitly supplied.

## Outputs

- `synthetic_benchmark.json` and `.csv`: raw query-level ranks and provenance;
- `summary_top_k.csv`: combined, outline, and ribbon Recall@K with Wilson CIs;
- `recall_at_k_1_150.csv` and `.png`: complete combined recall curves;
- `noise_ci_overlap.csv`: CI overlap and paired exact McNemar checks;
- `dose_response.png`: Recall@K versus 25%, 50%, and 75% visibility;
- `scaling_comparison.csv`: current index beside supplied baseline reports;
- `summary.md`: concise human-readable results table.

CI overlap is reported as a descriptive check. Because clean and noisy queries
are paired by parent, the benchmark also reports an exact paired McNemar p-value
at each requested K.
