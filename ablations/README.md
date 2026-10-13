# Stage 3 ablations

These scripts measure how much the learnt rankers contribute by replacing only
the selection step. Detection and recognition are not re-run: every variant
reads the same cached lines from `../artifacts/spatial_cache_test.pkl`, so the
input to Stage 3 is identical in all of them. They need no GPU and finish in
seconds.

The results were measured after the paper was accepted and are not part of it.

| Script | Stage 3 is replaced by | Needs | Writes |
|---|---|---|---|
| `04_ranker_ablation.py` | the regex selector from `../baselines/`, run unchanged | standard library | `results/ranker_ablation_results.csv` |
| `05_heuristic_strong.py` | a stronger hand-written selector, written once and not tuned | standard library | `results/ranker_ablation_v2_results.csv` |
| `06_feature_argmax_ablation.py` | the argmax of a single feature over the rankers' own candidates | `pandas`, `pyarrow` | `results/feature_argmax_ablation.csv` |

`04` and `06` do not re-implement anything they compare against. They read the
regex selector, the metric functions and the feature lists out of the original
source files and execute them as they are. `04` also recomputes the paper's
Table II from the stored predictions as a check, and prints both.

## Results

F1 / exact match on the 347 test receipts.

| Stage 3 selector | Company | Date | Address | Total | Average |
|---|---|---|---|---|---|
| Regex v1 | 75.86 / 52.16 | 91.55 / 91.35 | 77.12 / 18.16 | 39.77 / 39.77 | 71.07 / 50.36 |
| Heuristics v2 | 57.21 / 40.35 | 91.55 / 91.35 | 71.33 / 23.92 | 57.93 / 57.93 | 69.50 / 53.39 |
| Single-feature argmax | 86.37 / 58.21 | 96.57 / 96.36 | 69.52 / 19.60 | 69.74 / 69.74 | 80.55 / 60.98 |
| Learnt rankers | 89.09 / 61.38 | 92.12 / 91.93 | 83.96 / 25.07 | 84.15 / 84.15 | 87.33 / 65.63 |

## Reading the numbers

- **Regex v1** loses most on `total`. Its pattern matches the first `total`
  keyword on the receipt, including the one inside `SUBTOTAL`, and then takes
  the next number even when it is on another line.
- **Heuristics v2** fixes that and reaches 57.93 on `total`, but its rules for
  `company` and `address` do worse than v1. It was left as written rather than
  tuned against the test scores.
- **Single-feature argmax** is the strictest comparison, because it uses the
  rankers' candidates and features. The feature is chosen per field on the
  training split only: `has_comp_kw_below` (minimum) for `company`,
  `dist_to_kw_above` (minimum) for `date`, `comma_per_line` (maximum) for
  `address`, `has_cash_in_ctx` (maximum) for `total`.
- Against that baseline the rankers gain 14.44 F1 on `address` and 14.41 on
  `total`, 2.72 on `company`, and lose 4.45 on `date`. Date candidates are few,
  about 1.2 per receipt, and nearly all of them are already correct, so there is
  little left for a ranker to decide.
