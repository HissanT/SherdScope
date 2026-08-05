# Evaluation scores

## Run provenance

- Source: `outputs/matcher_v15_final_balanced_metadata`
- Created: `2026-08-04T16:16:30.619085+00:00`
- Shape matcher: `two-wall-joint-fgw-v15-final-balanced-metadata`
- Metadata model: `hesban-chip-metadata-v6-final-balanced`
- Evaluated queries: **29**. Query 10 is excluded because its stated true parent does not exist in the reference catalogue.
- Stored retrieval cutoff: **300** candidates. These are the latest completed results; they predate the later code default of 400.
- `NR` means the true parent did not reach that arm's fine-scored pool, so no final exact rank exists. A separately forced parent score is reported for cost diagnosis but is never converted into a fictional rank.
- Runtime cells use `total [retrieval/coarse/medium/fine]` in seconds.

## Aggregate results

| Arm | MRR | Top-1 | Top-5 | Top-10 | Fine-pool parent recall |
|---|---:|---:|---:|---:|---:|
| Shape only | 0.8448 | 24/29 (82.8%) | 25/29 (86.2%) | 25/29 (86.2%) | 25/29 (86.2%) |
| Shape + metadata | 0.8602 | 24/29 (82.8%) | 26/29 (89.7%) | 27/29 (93.1%) | 27/29 (93.1%) |

The shape-only 300-candidate retrieval pool retained the true parent for **27/29 (93.1%)** queries. Mean Top-5 overlap between arms was **3.93/5**. Shape-only runtime totaled **3.92 h**; the independent metadata-aware arm totaled **3.88 h**.

## A. Final ranking and shape-cost diagnosis

| Query | True parent | Shape rank | S@1 | S@5 | S@10 | Fused rank | M@1 | M@5 | M@10 | Shape winner | Metadata winner | Parent cost | Winner cost | Gap |
|---:|---|---:|:---:|:---:|:---:|---:|:---:|:---:|:---:|---|---|---:|---:|---:|
| Q1 | 3.1.8 | 1 | Y | Y | Y | 1 | Y | Y | Y | 3.1.8 | 3.1.8 | 0.154872 | 0.154872 | 0.000000 |
| Q2 | 3.2.4 | 1 | Y | Y | Y | 1 | Y | Y | Y | 3.2.4 | 3.2.4 | 0.200918 | 0.200918 | 0.000000 |
| Q3 | 3.3.6 | NR | N | N | N | 9 | N | N | Y | 3.3.7 | 3.3.7 | 0.207983 | 0.163271 | 0.044712 |
| Q4 | 3.4.9 | 1 | Y | Y | Y | 1 | Y | Y | Y | 3.4.9 | 3.4.9 | 0.152619 | 0.152619 | 0.000000 |
| Q5 | 3.5.8 | 1 | Y | Y | Y | 1 | Y | Y | Y | 3.5.8 | 3.5.8 | 0.151233 | 0.151233 | 0.000000 |
| Q6 | 3.6.10 | 1 | Y | Y | Y | 1 | Y | Y | Y | 3.6.10 | 3.6.10 | 0.203624 | 0.203624 | 0.000000 |
| Q7 | 3.7.7 | 1 | Y | Y | Y | 1 | Y | Y | Y | 3.7.7 | 3.7.7 | 0.184258 | 0.184258 | 0.000000 |
| Q8 | 3.8.5 | 1 | Y | Y | Y | 1 | Y | Y | Y | 3.8.5 | 3.8.5 | 0.161571 | 0.161571 | 0.000000 |
| Q9 | 3.9.8 | 2 | N | Y | Y | 2 | N | Y | Y | 3.9.10 | 3.9.10 | 0.175249 | 0.156447 | 0.018802 |
| Q11 | 3.12.5 | 1 | Y | Y | Y | 1 | Y | Y | Y | 3.12.5 | 3.12.5 | 0.210992 | 0.210992 | 0.000000 |
| Q12 | 3.13.6 | 1 | Y | Y | Y | 1 | Y | Y | Y | 3.13.6 | 3.13.6 | 0.188247 | 0.188247 | 0.000000 |
| Q13 | 3.14.10 | 1 | Y | Y | Y | 1 | Y | Y | Y | 3.14.10 | 3.14.10 | 0.132340 | 0.132340 | 0.000000 |
| Q14 | 3.15.4 | 1 | Y | Y | Y | 1 | Y | Y | Y | 3.15.4 | 3.15.4 | 0.115228 | 0.115228 | 0.000000 |
| Q15 | 3.16.1 | 1 | Y | Y | Y | 1 | Y | Y | Y | 3.16.1 | 3.16.1 | 0.122329 | 0.122329 | 0.000000 |
| Q16 | 3.17.9 | 1 | Y | Y | Y | 1 | Y | Y | Y | 3.17.9 | 3.17.9 | 0.163920 | 0.163920 | 0.000000 |
| Q17 | 3.18.5 | 1 | Y | Y | Y | 1 | Y | Y | Y | 3.18.5 | 3.18.5 | 0.091734 | 0.091734 | 0.000000 |
| Q18 | 3.19.5 | 1 | Y | Y | Y | 1 | Y | Y | Y | 3.19.5 | 3.19.5 | 0.192339 | 0.192339 | 0.000000 |
| Q19 | 3.20.20 | NR | N | N | N | NR | N | N | N | 3.47.11 | 3.21.24 | 0.308352 | 0.275244 | 0.033109 |
| Q20 | 3.21.16 | 1 | Y | Y | Y | 1 | Y | Y | Y | 3.21.16 | 3.21.16 | 0.167931 | 0.167931 | 0.000000 |
| Q21 | 3.22.30 | 1 | Y | Y | Y | 1 | Y | Y | Y | 3.22.30 | 3.22.30 | 0.193498 | 0.193498 | 0.000000 |
| Q22 | 3.23.2 | 1 | Y | Y | Y | 1 | Y | Y | Y | 3.23.2 | 3.23.2 | 0.121968 | 0.121968 | 0.000000 |
| Q23 | 3.24.18 | 1 | Y | Y | Y | 1 | Y | Y | Y | 3.24.18 | 3.24.18 | 0.189300 | 0.189300 | 0.000000 |
| Q24 | 3.25.4 | 1 | Y | Y | Y | 1 | Y | Y | Y | 3.25.4 | 3.25.4 | 0.197004 | 0.197004 | 0.000000 |
| Q25 | 3.26.12 | 1 | Y | Y | Y | 1 | Y | Y | Y | 3.26.12 | 3.26.12 | 0.145134 | 0.145134 | 0.000000 |
| Q26 | 3.27.19 | 1 | Y | Y | Y | 1 | Y | Y | Y | 3.27.19 | 3.27.19 | 0.143424 | 0.143424 | 0.000000 |
| Q27 | 3.29.4 | NR | N | N | N | 3 | N | Y | Y | 3.13.8 | 3.65.3 | 0.167394 | 0.155456 | 0.011938 |
| Q28 | 3.43.4 | 1 | Y | Y | Y | 1 | Y | Y | Y | 3.43.4 | 3.43.4 | 0.171039 | 0.171039 | 0.000000 |
| Q29 | 3.47.2 | NR | N | N | N | NR | N | N | N | 3.47.1 | 3.46.27 | 0.246705 | 0.210167 | 0.036538 |
| Q30 | 3.81.5 | 1 | Y | Y | Y | 1 | Y | Y | Y | 3.81.5 | 3.81.5 | 0.161609 | 0.161609 | 0.000000 |

## B. Retrieval and cascade diagnosis (shape-only arm)

| Query | In 300 pool? | Retrieval rank | Outline rank | Ribbon rank | In fine pool? |
|---:|:---:|---:|---:|---:|:---:|
| Q1 | Y | 3 | 2 | 10 | Y |
| Q2 | Y | 1 | 1 | 28 | Y |
| Q3 | Y | 34 | 19 | 91 | N |
| Q4 | Y | 38 | 22 | 37 | Y |
| Q5 | Y | 9 | 9 | 5 | Y |
| Q6 | Y | 185 | 105 | 115 | Y |
| Q7 | Y | 5 | 119 | 3 | Y |
| Q8 | Y | 71 | 40 | 83 | Y |
| Q9 | Y | 54 | 30 | 211 | Y |
| Q11 | Y | 13 | 7 | 8 | Y |
| Q12 | Y | 9 | 5 | 189 | Y |
| Q13 | Y | 3 | 13 | 2 | Y |
| Q14 | Y | 1 | 1 | 1 | Y |
| Q15 | Y | 2 | 334 | 1 | Y |
| Q16 | Y | 1 | 1 | 192 | Y |
| Q17 | Y | 1 | 1 | 1 | Y |
| Q18 | Y | 14 | 63 | 7 | Y |
| Q19 | N | >300 | 227 | 306 | N |
| Q20 | Y | 1 | 8 | 1 | Y |
| Q21 | Y | 3 | 2 | 104 | Y |
| Q22 | Y | 2 | 1 | 71 | Y |
| Q23 | Y | 4 | 3 | 9 | Y |
| Q24 | Y | 5 | 3 | 3 | Y |
| Q25 | Y | 2 | 1 | 272 | Y |
| Q26 | Y | 1 | 1 | 3 | Y |
| Q27 | Y | 4 | 184 | 2 | N |
| Q28 | Y | 2 | 459 | 1 | Y |
| Q29 | N | >300 | 880 | 335 | N |
| Q30 | Y | 48 | 380 | 25 | Y |

## C. Paired metadata effect

`Metadata shape rank` and `fused rank` are calculated inside the same metadata-aware fine pool; positive change means metadata improved the parent's rank.

| Query | Metadata shape rank | Fused rank | Change | Direction | Parent shape cost | Metadata cost | Fused cost | Fields | Top-5 overlap |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| Q1 | 1 | 1 | +0 | unchanged | 0.154872 | 0.101127 | 0.148034 | 4 | 3/5 |
| Q2 | 1 | 1 | +0 | unchanged | 0.200918 | 0.122338 | 0.194663 | 4 | 4/5 |
| Q3 | 11 | 9 | +2 | improved | 0.213491 | 0.059816 | 0.201923 | 3 | 4/5 |
| Q4 | 1 | 1 | +0 | unchanged | 0.152619 | 0.058605 | 0.140902 | 3 | 4/5 |
| Q5 | 1 | 1 | +0 | unchanged | 0.151233 | 0.179386 | 0.146545 | 4 | 3/5 |
| Q6 | 1 | 1 | +0 | unchanged | 0.203624 | 0.024948 | 0.187766 | 3 | 3/5 |
| Q7 | 1 | 1 | +0 | unchanged | 0.184258 | 0.284646 | 0.182527 | 3 | 5/5 |
| Q8 | 1 | 1 | +0 | unchanged | 0.161571 | 0.036615 | 0.147149 | 3 | 4/5 |
| Q9 | 2 | 2 | +0 | unchanged | 0.175249 | 0.076372 | 0.165718 | 3 | 5/5 |
| Q11 | 1 | 1 | +0 | unchanged | 0.210992 | 0.100134 | 0.204371 | 3 | 5/5 |
| Q12 | 1 | 1 | +0 | unchanged | 0.188247 | 0.044131 | 0.174749 | 3 | 4/5 |
| Q13 | 1 | 1 | +0 | unchanged | 0.132340 | 0.051415 | 0.119738 | 3 | 5/5 |
| Q14 | 1 | 1 | +0 | unchanged | 0.115228 | 0.031352 | 0.100159 | 3 | 5/5 |
| Q15 | 1 | 1 | +0 | unchanged | 0.122329 | 0.071581 | 0.111836 | 4 | 2/5 |
| Q16 | 1 | 1 | +0 | unchanged | 0.163920 | 0.061506 | 0.152141 | 4 | 4/5 |
| Q17 | 1 | 1 | +0 | unchanged | 0.091734 | 0.026964 | 0.075549 | 4 | 4/5 |
| Q18 | 1 | 1 | +0 | unchanged | 0.192339 | 0.072373 | 0.182316 | 3 | 1/5 |
| Q19 | NR | NR | — | NR | — | — | — | — | 4/5 |
| Q20 | 1 | 1 | +0 | unchanged | 0.167931 | 0.042023 | 0.153668 | 4 | 4/5 |
| Q21 | 1 | 1 | +0 | unchanged | 0.193498 | 0.046468 | 0.180288 | 3 | 4/5 |
| Q22 | 1 | 1 | +0 | unchanged | 0.121968 | 0.069895 | 0.111260 | 4 | 5/5 |
| Q23 | 1 | 1 | +0 | unchanged | 0.189300 | 0.159552 | 0.184068 | 4 | 4/5 |
| Q24 | 1 | 1 | +0 | unchanged | 0.197004 | 0.039567 | 0.182427 | 4 | 5/5 |
| Q25 | 1 | 1 | +0 | unchanged | 0.145134 | 0.041011 | 0.131252 | 3 | 4/5 |
| Q26 | 1 | 1 | +0 | unchanged | 0.143424 | 0.046126 | 0.130171 | 3 | 4/5 |
| Q27 | 4 | 3 | +1 | improved | 0.167802 | 0.013648 | 0.150554 | 3 | 3/5 |
| Q28 | 1 | 1 | +0 | unchanged | 0.171039 | 0.041546 | 0.156715 | 4 | 3/5 |
| Q29 | NR | NR | — | NR | — | — | — | — | 4/5 |
| Q30 | 1 | 1 | +0 | unchanged | 0.161609 | 0.036475 | 0.147170 | 3 | 5/5 |

## D. Runtime by query

| Query | Shape seconds: total [R/C/M/F] | Metadata seconds: total [R/C/M/F] | Combined total (s) |
|---:|---:|---:|---:|
| Q1 | 579.0 [1.1/349.8/57.7/170.4] | 582.6 [1.2/349.3/57.4/174.7] | 1161.6 |
| Q2 | 585.4 [1.1/351.3/60.2/172.9] | 604.1 [1.1/371.7/58.9/172.4] | 1189.5 |
| Q3 | 585.4 [0.9/356.5/59.3/168.8] | 597.2 [1.0/356.2/59.9/180.0] | 1182.7 |
| Q4 | 567.7 [1.2/319.5/62.6/184.4] | 572.6 [1.3/325.9/62.3/183.2] | 1140.4 |
| Q5 | 559.7 [1.2/311.2/62.8/184.4] | 553.9 [1.6/308.2/62.1/182.0] | 1113.5 |
| Q6 | 555.6 [1.2/315.0/60.4/178.9] | 556.6 [1.3/314.9/60.6/179.8] | 1112.2 |
| Q7 | 554.8 [1.3/313.6/62.9/177.0] | 557.2 [1.3/317.4/61.6/177.0] | 1112.0 |
| Q8 | 569.8 [1.2/317.4/62.0/189.3] | 578.9 [1.3/322.5/61.2/193.8] | 1148.7 |
| Q9 | 544.8 [0.9/315.3/56.6/172.0] | 546.1 [1.2/311.6/57.7/175.5] | 1090.8 |
| Q11 | 565.3 [1.0/326.5/60.3/177.5] | 641.8 [1.3/334.9/66.0/239.6] | 1207.1 |
| Q12 | 610.9 [1.4/382.6/56.9/170.0] | 584.9 [1.3/357.5/57.6/168.4] | 1195.8 |
| Q13 | 611.3 [0.9/331.0/69.8/209.6] | 524.2 [1.1/292.0/54.2/176.8] | 1135.5 |
| Q14 | 578.4 [1.2/348.6/58.1/170.6] | 574.4 [1.1/346.8/57.5/169.0] | 1152.8 |
| Q15 | 532.2 [0.9/353.2/58.9/119.1] | 330.6 [0.9/204.4/31.8/93.6] | 862.8 |
| Q16 | 320.3 [0.9/196.1/31.8/91.5] | 331.0 [0.9/201.7/33.0/95.4] | 651.3 |
| Q17 | 332.0 [0.9/207.0/32.6/91.5] | 322.7 [0.9/198.8/32.2/90.9] | 654.7 |
| Q18 | 324.8 [0.9/197.2/33.6/93.2] | 327.5 [0.8/195.7/34.4/96.6] | 652.3 |
| Q19 | 338.4 [0.9/208.0/33.7/95.8] | 323.0 [1.0/196.1/33.1/92.7] | 661.3 |
| Q20 | 304.0 [0.9/180.2/31.2/91.7] | 307.9 [0.9/182.8/31.4/92.8] | 612.0 |
| Q21 | 318.9 [0.9/194.4/31.0/92.6] | 317.4 [1.0/192.5/31.5/92.4] | 636.3 |
| Q22 | 336.2 [0.9/207.0/32.1/96.1] | 330.0 [1.0/199.8/32.9/96.3] | 666.1 |
| Q23 | 499.9 [0.8/250.8/68.1/180.3] | 590.4 [1.4/356.4/58.4/174.3] | 1090.3 |
| Q24 | 552.3 [0.9/350.2/40.9/160.4] | 559.4 [1.1/360.0/40.8/157.6] | 1111.7 |
| Q25 | 562.7 [0.9/334.5/58.3/169.0] | 563.4 [1.1/329.7/60.1/172.5] | 1126.1 |
| Q26 | 584.3 [0.9/355.4/58.5/169.5] | 363.9 [1.1/231.2/35.3/96.3] | 948.2 |
| Q27 | 328.6 [0.9/199.7/33.8/94.2] | 328.6 [0.8/200.0/33.4/94.4] | 657.2 |
| Q28 | 348.6 [0.8/199.4/33.7/114.7] | 336.1 [0.9/208.4/32.0/94.8] | 684.7 |
| Q29 | 336.7 [0.7/195.2/33.5/107.3] | 456.2 [1.0/228.7/40.3/186.2] | 792.9 |
| Q30 | 633.5 [1.8/373.2/64.0/194.4] | 711.0 [1.5/417.5/73.9/218.1] | 1344.5 |

## Metric definitions

- **Parent exact rank:** position of the true parent in the arm's final fine-scored pool. Missing parents are `NR` and contribute zero to MRR and Top-K.
- **Parent/winner gap:** forced-or-in-pool true-parent shape cost minus the returned rank-1 shape cost. Zero means the parent won; a small positive value is a near miss.
- **Retrieval rank:** combined cheap-retrieval rank when retained. `>300` means it was outside the stored pool; outline and ribbon ranks remain shown independently.
- **Fine pool:** whether the parent survived retrieval, coarse scoring, and medium scoring to receive the final 96-sample score.
- **Metadata rank change:** metadata-aware fine-pool shape rank minus fused rank. This isolates reranking direction without pretending the two independent arms shared an identical pool.
- **Top-5 overlap:** number of reference IDs shared by the shape-only and metadata-aware Top 5.
