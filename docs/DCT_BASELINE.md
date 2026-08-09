# Discrete Cosine Transform baseline

This baseline gives SherdScope a simple, independent shape-retrieval method for
comparison with the production two-wall FGW matcher. It follows the main DCT
choices reported by Wilczek et al. in *A computer tool to identify best matches
for pottery fragments*: a rim outline is represented by 100 points, 50 from
each wall, and reduced to 20 DCT harmonics. The baseline is deliberately
shape-only. It does not use metadata and it does not call the production
matcher.

## What DCT does

A pottery profile can be treated as two ordered signals: its horizontal and
vertical coordinates as we travel from one fracture edge, around the rim, and
down the other wall. DCT rewrites each coordinate signal as a sum of smooth
cosine waves. The first waves describe broad form. Later waves describe smaller
bends and details. Keeping only the first 20 harmonics produces a short,
repeatable description of the profile.

For coordinate value `x[n]`, the orthonormal type-II DCT is:

```text
C[k] = alpha[k] * sum from n=0 to N-1 of
       x[n] * cos(pi/N * (n + 1/2) * k)
```

The calculation is performed separately for horizontal and vertical
coordinates. A query and reference are compared with root-mean-square
difference across their retained coefficient pairs:

```text
D(query, reference) = sqrt(mean((C_query - C_reference)^2))
```

A smaller number means the two shapes are more similar.

## SherdScope preprocessing

The reviewed rim point is placed at the origin. The average direction from the
rim toward the fracture ends is rotated onto a common vertical axis. The
outline is divided by its root-mean-square radius to remove drawing size. This
is one deterministic preprocessing operation, not an iterative alignment.

Real fragments preserve unknown amounts of a complete reference. For that
reason, the baseline tests 11 reference lengths from 50% through 100% and keeps
the lowest DCT distance for each reference. Values below 50% are deliberately
excluded because the reviewed queries already represent substantial pottery
chunks; allowing smaller reference pieces would create unrealistically easy
local matches. This coverage search is the main adaptation needed for
SherdScope. The chosen coverage is saved with every result so that the decision
remains inspectable.

## Default hyperparameters

`samples = 100` means that every outline contains 100 ordered points. Fifty
come from each pottery wall. More points preserve finer geometry but make the
descriptor more sensitive to small extraction noise.

`harmonics = 20` controls how many cosine waves are retained for each
coordinate. This is the value used in the paper. Fewer harmonics produce a
smoother and less detailed description. More harmonics preserve small details,
but may also preserve scanning noise and jagged mask edges.

`min_reference_coverage = 0.50` and `max_reference_coverage = 1.00` define the
shortest and longest portion of a complete reference that may be compared with
the fragment. The 50% floor prevents a candidate from winning by matching an
unrealistically small, easy section.

`coverage_steps = 11` tests 50%, 55%, 60%, and so on through 100%. More steps
give a finer length search but increase runtime and can make the experiment
more flexible.

`include_dc = true` keeps the zero-frequency coefficient. Translation has
already been removed by placing the rim at the origin, so this coefficient
retains useful information about where the complete outline lies relative to
the rim. `--exclude-dc` is available as an explicit ablation.

## Score one query

Run the command from the SherdScope directory. A query can be an existing
SherdScope query identifier:

```powershell
python -m scripts.matcher.dct_baseline score PROJECT_PATH `
  --query-id QUERY_ID `
  --output outputs/dct_query_run
```

It can also be a saved contour artifact:

```powershell
python -m scripts.matcher.dct_baseline score PROJECT_PATH `
  --query-artifact path/to/artifact.json `
  --output outputs/dct_query_run
```

The output contains `dct_results.json`, `dct_results.csv`, and
`query_dct_reconstruction.png`. The image shows the normalized 100-point query
beside its 20-harmonic reconstruction.

## Run known-parent query manifests

The batch command accepts one or more existing matcher manifests, verifies each
saved query artifact against its recorded SHA-256 hash, resolves the true
Figure and Item to exactly one reference, builds one shared DCT reference bank,
and scores queries in parallel. For the existing 29-query and 40-query cohorts:

```powershell
python -m scripts.matcher.dct_baseline batch `
  projects/Finalized_Hesban_Corpus_Digitization_20260805_154332 `
  --manifest outputs/development_29_2626_pool400_20260807/batch_manifest.json `
  --manifest outputs/hesban_40_2626_pool400_20260807/batch_manifest.json `
  --output outputs/dct_known_parent_69_coverage50_20260808 `
  --workers 8
```

The batch writes `dct_batch.json`, `dct_batch.csv`, and `summary.md`. Results
include the exact true-parent rank, DCT distance, selected reference coverage,
Top-1 candidate, saved Top-20 candidates, per-query runtime, combined metrics,
and separate metrics for both cohorts. The terminal prints when the shared
reference bank starts and finishes, then prints one progress line for every
completed query, including the completed count, true-parent rank, selected
coverage, query time, and total elapsed time.

## Run the synthetic baseline

```powershell
python -m scripts.matcher.dct_baseline synthetic PROJECT_PATH `
  --output outputs/dct_synthetic_300 `
  --sample-size 300
```

The experiment uses the same deterministic clean, noisy, and partial-fragment
conditions as SherdScope's existing synthetic benchmark. It writes raw JSON
and CSV records plus a Markdown summary containing Top-1, Top-3, Top-5,
Top-10, median rank, and mean reciprocal rank.

The synthetic experiment keeps the known parent in the reference library and
therefore measures exact-parent retrieval. It is not the same as the paper's
class-level leave-one-out experiment. Results from the two datasets must not be
treated as directly comparable percentages.

Existing output files are protected. Use `--overwrite-output` only when the
selected DCT result directory is intentionally being replaced. Reviewed masks,
contour artifacts, saved queries, and production matcher runs are never
modified.
