# 2026-08-03 - SherdScope Research Evidence Checkpoint

## Purpose

This checkpoint freezes the evidence collected before the next real-query matching phase. It separates measured results, user-reported effort, software validation, and interpretation. It also records failed or superseded experiments so that later writing does not present only the best result.

## Data state and evaluation rules

- The working corpus reached approximately 1,099 reviewed reference images as a project milestone. The v9-v11 matcher runs used exactly 1,058 indexed, matcher-eligible reference contours.
- Thirty synthetic partial-profile queries were prepared from known Hesban parents. Query 10 is an open-set exception because its nominal parent does not exist in the eligible reference library. Closed-set accuracy therefore uses 29 queries.
- Query 6 contains an export/display artifact: an extra separately searched row was saved even though the target already appeared in the ranked results. It is ignored when counting Top-5 accuracy.
- Exact success means the correct Hesban Figure + Item, not merely a similar form or the correct figure.
- The same 29 closed-set queries were used during development. Comparisons between versions are diagnostic development comparisons, not independent held-out estimates.

## Exact-item matcher comparison

| Matcher | Top-1 | Top-5 | Missed targets | Wall time for 30 runs |
|---|---:|---:|---|---:|
| v9 shape only | 22/29 (75.86%) | 23/29 (79.31%) | Q3, Q19, Q22, Q25, Q27, Q29 | 15,778.45 s (4 h 22 m 58 s) |
| v10 shape only | 18/29 (62.07%) | 18/29 (62.07%) | Q3, Q4, Q6, Q8, Q9, Q11, Q12, Q18, Q19, Q29, Q30 | 10,082.32 s (2 h 48 m 2 s) |
| v11 shape only | 24/29 (82.76%) | 26/29 (89.66%) | Q3, Q19, Q29 | 11,778.60 s (3 h 16 m 19 s) |
| v11 shape + metadata | 20/29 (68.97%) | 25/29 (86.21%) | Q3, Q11, Q19, Q29 | Same shape run; metadata reranks the shared fine-scored pool |

V10 was faster but substantially worse. Its more aggressive persistent-local retrieval and cascade reservations caused many known parents to disappear before final scoring. Seven of its 11 closed-set misses were pruned at the medium stage, one at coarse, and three before coarse. V11 therefore returned to the validated two-channel outline/ribbon retrieval path, preserved ordinary score-ranked candidates, and appended only bounded channel champions. This restored or improved most v9 successes without returning fully to the v9 runtime.

For v11, the exact target was present in the shape candidate pool for 26/29 evaluable queries. Conditional on presence, shape ranked all 26 within the Top-5 and 24/26 at rank one. The three remaining failures were retrieval failures: Q3 had combined retrieval rank 324, Q19 rank 352, and Q29 rank 430, outside the 300-candidate pool. Their outline/ribbon ranks were 195/258, 222/277, and 782/284 respectively. Their separately forced shape scores did not beat the v11 rank-one result, so these cases reflect both retrieval weakness and imperfect final discrimination rather than a hidden guaranteed Top-1 result.

V11 runtime remained too high for corpus growth. The measured internal mean was 385.90 seconds per query and the median was 363.48 seconds. The fastest query took 277.20 seconds and the slowest 628.27 seconds. Mean stage times were 1.57 seconds for cheap retrieval, 231.07 seconds for coarse scoring, 39.21 seconds for medium scoring, and 114.06 seconds for fine scoring. The coarse geometric pass is therefore the dominant target for optimization. At the same speed, increasing the library from roughly 1,058 to 4,000-5,000 references would be impractical unless candidate scoring is reduced, cached, vectorized, or parallelized more effectively.

## First metadata-fusion result

Only shape-only and shape-plus-metadata are scientifically meaningful comparisons. A metadata-only identity experiment is not included because broad facts such as diameter or colour cannot identify a vessel without morphological evidence.

The first fusion run used up to four observations available from the query workflow: rim diameter and exterior, interior, and core Munsell colours. Missing fields were neutral. Numeric and colour differences contributed gradual costs rather than hard thresholds, and shape retained the larger weight. The mean metadata weight for targets in the shape pool was 0.1381. The mean target metadata cost was 0.4913 and the median was 0.5478. Fifteen targets had three compared fields and 11 had four.

Metadata improved the target rank for Q9 and Q27, left 18 targets unchanged, and worsened Q2, Q4, Q5, Q11, Q26, and Q30. It moved Q11 from shape rank 1 to combined rank 8, removing a correct target from the Top-5. Across all queries, the mean overlap between shape and shape-plus-metadata Top-5 lists was 3.33 candidates. The result is useful because it proves that the fusion mechanism is active, but its first calibration is not yet beneficial overall: Top-1 fell from 82.76% to 68.97%, and Top-5 fell from 89.66% to 86.21%.

The next metadata experiment must be tuned only on a development split and evaluated on new queries. It should retain the continuous, uncertainty-aware score and blank-field neutrality while testing lower influence, reliability calibration, and field-specific noise models. Metadata may move a shape-plausible candidate out of the final shortlist when several fields disagree, but it must not introduce a reference rejected by the shape candidate pool.

## Synthetic matching diagnostics

The first 360-query synthetic generator was rejected after visual inspection because it sometimes treated the fracture cap as the archaeological rim. Those values remain a software audit, not valid archaeological performance.

After correcting the rim selection, 90 parent references were each tested under four conditions (360 queries total): grid-clean, clean partial, light noise, and moderate noise. The grid-clean condition was intentionally easy and reached 68/90 Top-1 (75.56%) and 90/90 Top-5. The realistic partial conditions were much harder:

| Corrected condition | Retrieval Top-1 | Top-5 | Top-10 | Top-150 |
|---|---:|---:|---:|---:|
| Clean partial | 22/90 (24.44%) | 47/90 (52.22%) | 57/90 (63.33%) | 83/90 (92.22%) |
| Light noise | 23/90 (25.56%) | 47/90 (52.22%) | 58/90 (64.44%) | 82/90 (91.11%) |
| Moderate noise | 20/90 (22.22%) | 46/90 (51.11%) | 62/90 (68.89%) | 84/90 (93.33%) |

The surprising similarity between clean and noisy conditions shows that partial-profile selection and rim/fracture representation dominate small added point noise. V9's 300-candidate clean retrieval retained 87/90 parents (96.67%), versus 83/90 (92.22%) at the earlier 150-candidate boundary. A six-query full v8 rerank placed all six parents first, but n=6 is far too small for an accuracy claim. These experiments diagnose the retrieval-to-reranking pipeline; they do not estimate performance on independently photographed real sherds.

## Real-photograph segmentation benchmark

The original U-Net was trained from 68 labelled photographs. Its historical three-image holdout reported mean IoU 0.9327 and Dice 0.9651, but n=3 and narrow capture conditions make that result optimistic and unstable.

Five new photographs were manually outlined without showing the predictions first. All methods were compared against these same gold masks at 640 x 640 with a 3-pixel boundary tolerance. SAM 2.1 Tiny was prompted only from U-Net geometry, so it is not an independent manual-prompt benchmark. DeepLabV3+ was trained on the same separate 68-image source set.

| Method | Mean Dice | Mean IoU | Boundary F1@3 px | HD95 (px) | Mean surface distance (px) |
|---|---:|---:|---:|---:|---:|
| U-Net original | 0.9077 | 0.8362 | 0.6904 | 56.03 | 10.89 |
| U-Net + enclosed-hole filling | **0.9350** | **0.8829** | **0.7640** | **29.31** | **5.44** |
| SAM 2.1 Tiny, U-Net prompt | 0.9176 | 0.8535 | 0.6298 | 37.62 | 7.42 |
| DeepLabV3+ | 0.7747 | 0.6609 | 0.5051 | 47.13 | 10.52 |
| DeepLabV3+ + enclosed-hole filling | 0.7913 | 0.6856 | 0.5191 | 46.18 | 9.62 |

Topology-safe enclosed-hole filling was the best tested automatic option on this five-image sample. It repaired the central U-Net holes without changing the outer contour and improved mean Dice by 0.0273, IoU by 0.0466, boundary F1 by 0.0736, and HD95 by 26.73 pixels. SAM was competitive in region overlap but had worse boundary F1 than the repaired U-Net. DeepLab underperformed substantially. Image-level variation remained large, especially where glove, shadow, depth order, and the broken rear surface were ambiguous. Manual outlines therefore remain the query gold standard for the next matcher test.

Simple contour smoothing did not solve the problem by itself. In the five-photo pilot, raw-to-smoothed U-Net Dice changed only from 0.9072 to 0.9077, while boundary F1 changed from 0.6774 to 0.6857. Smoothing is appropriate for pixel stair-steps, but it cannot decide which depth surface is archaeologically relevant.

## Earlier validated subsystems retained in the paper

- The 150-vessel Hesban OCR development evaluation improved from 2,966/3,150 correct cells (94.16%) and 10/150 completely correct vessels (6.67%) in Eval1 to 3,142/3,150 cells (99.75%) and 144/150 complete vessels (96.00%) in Eval4. Eval4 had eight failed cells, two missing values, and 100% Firing accuracy.
- A roughly 50-profile segmentation pilot found that about half of automatic publication-profile masks could be accepted directly; the rest generally needed minor edits averaging about 10 seconds, and none required complete reconstruction. This is a workflow observation, not pixel accuracy.
- The user-reported full review effort was approximately four hours for roughly 1,100 vessel profiles, or about 13 seconds per vessel on average including navigation and decisions. This was not instrumented automatically.
- Approximately 350 accepted masks were curated for the experimental U-Net training workflow. Because these came from the same developing corpus, future splits must prevent near-duplicate or same-figure leakage.
- The ten-query multimodal LLM baseline reached at most 40% joint form-period Top-1 with metadata, but only 1/10 exact Figure + Item Top-5. This remains a weak baseline rather than the final method.

## Exploratory type-voting test

Catalogue type labels were propagated from ranked shape neighbors for the 29 queries with known targets. Because these synthetic queries originate from catalogue parents, allowing the exact parent to vote is circular: broad-family Top-1 was 24/29 (82.76%), essentially reflecting exact retrieval.

More conservative tests removed either the exact parent or every item from the target figure. With bucket-weighted voting, broad-family accuracy peaked at 21/29 (72.41%) for k=10 after removing only the parent, and 19/29 (65.52%) for k=10 after removing the entire target figure. Fine normalized-type accuracy was much weaker; after target-figure removal it peaked at 13/29 (44.83%). This suggests that neighbor voting may support coarse vessel-family classification, but the current result is exploratory, uses the same development queries, and depends on incomplete/noisy catalogue type labels.

## Major software and workflow changes now in place

1. Vessel identity moved from merged page masks to stable, independently reviewable detection records and reviewed bounding boxes.
2. High-DPI source-page crops, profile-mask review, brush/eraser/zoom tools, keyboard approval, and non-destructive accepted-mask storage were added.
3. Hesban table extraction became geometry-first and evidence-preserving, with exact row/column overlays, retries, reviewer overrides, and 99.75% development cell accuracy.
4. Complementary PDFs can be linked as one combined corpus. Exact cross-PDF duplicate figures are collapsed only with matching vessel-number evidence.
5. Decimal-containing filenames and legacy figure-key collisions were fixed so Match Query shows Figure + Item names instead of cryptic card IDs.
6. Autosave/revision races, stuck processing labels, warning overrides, long sidebars, and review-workspace sizing were corrected.
7. Approved profile masks, vessel boxes, and linkage records are preserved on ordinary reruns. Force/reset actions remain intentionally destructive and must be used only when regeneration is desired.
8. Canonical contours, a 30-query annotation curator, resumable batch matching, isolated workbooks, forced known-target diagnostics, v9-v11 versioning, candidate lineage, and per-stage timing were added.
9. Metadata fusion was implemented as a continuous, uncertainty-aware score over the same shape-plausible pool, with missing fields neutral and per-field explanations stored.
10. A five-photo real-shard pilot and segmentation benchmark now preserve gold masks separately and compare U-Net, topology repair, SAM, and DeepLab without overwriting manual labels.

## Threats to validity and next decision

The 29 closed-set queries have been reused throughout development, so v11's 82.76% Top-1 and 89.66% Top-5 are development performance and may overfit these parents, annotations, and corpus conventions. The queries are synthetic partial profiles cut from their own catalogue references, not independent real sherds. The five-photo segmentation comparison is also too small for model selection beyond a pilot.

Before further formula tuning, the next phase should lock the current v11 shape matcher and test manually outlined real sherds. Shape-plus-metadata should be evaluated beside shape-only using the same candidate pool, but not tuned on the final real test cases. Runtime optimization should target coarse scoring and must be checked for recall loss. Every future report should preserve exact ranks, candidate-stage lineage, runtime, dataset size, exceptions, and segmentation source so that an apparent accuracy improvement can be traced to its actual cause.

## Synchronized records

- AAAI-style source and compiled paper: `outputs/progress-report/SherdScope_AAAI_Style_Paper.tex` and `outputs/progress-report/SherdScope_AAAI_Style_Paper.pdf`
- Notion checkpoint: https://app.notion.com/p/3b1b46e01407812faa1ffe90cb0d20fa
