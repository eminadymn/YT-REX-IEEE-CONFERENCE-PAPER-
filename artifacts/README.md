# Artefacts of the paper run

Intermediate outputs of the Kaggle T4 run that produced the paper's numbers. The
full log of that run is `../results/kernel_log.txt`, and the candidate and
positive counts below match it line for line.

With these files everything after recognition can be re-run on a CPU. See
"Without a GPU" in the top-level README.

## `spatial_cache_{train,test}.pkl`

Output of detection, line merging and recognition: 626 training receipts with
30,694 lines, and 347 test receipts with 17,018 lines.

```python
{
  "X00016469670": {                 # image id, the file name without extension
    "filename": "X00016469670.jpg",
    "spatial_lines": [              # sorted top to bottom
      {
        "text": "TAN CHAY YEE",
        "x1": 99, "y1": 27, "x2": 311, "y2": 63,     # pixels
        "rel_x": 0.443, "rel_y": 0.050,             # box centre / image size
        "w_ratio": 0.458, "h_ratio": 0.040,         # box size / image size
        "x1_ratio": 0.214, "x2_ratio": 0.672,
      },
      ...
    ],
  },
  ...
}
```

The files are standard Python pickles holding only dicts, lists, strings,
integers and floats. As with any pickle, load them only from a source you trust.

## `candidates_{field}_{split}.parquet`

One row per candidate, written by `01_data_preparation.py`. Every table has
`base` (image id), `value_str` (the candidate text), `label` (1 for a positive)
and the feature columns listed under `FEATURES` in `02_model_training.py`, plus
a few bookkeeping columns such as `line_idx`.

| Field | Train rows | Train positives | Test rows | Test positives |
|---|---|---|---|---|
| `total` | 7,156 | 1,549 | 4,100 | 807 |
| `company` | 16,504 | 592 | 9,197 | 326 |
| `date` | 760 | 727 | 405 | 386 |
| `address` | 66,118 | 612 | 37,024 | 337 |

Not every receipt has a date candidate: 605 of 626 training receipts and 330 of
347 test receipts do.

## `ytrex_predictions.csv`

The predictions behind Table II, one row per test receipt: `pred_<field>`,
`gt_<field>`, and `<field>_f1`, `<field>_em`, `<field>_anls` for each of the
four fields. Read it with `dtype=str` so that values such as `193.00` keep their
trailing zeros.

## `split_meta.json`

The sorted image ids of both splits. `02_model_training.py` reads it for the
order of the test set. This file was not downloaded from the run: it was
regenerated from the keys of the two caches, which is how
`01_data_preparation.py` builds it.

## Not included

- `ranker_<field>.pkl`, the fitted rankers. `02_model_training.py` refits them
  from the candidate tables in under a minute.
- `test_timings.json`, the per-image timings. The aggregated latencies are in
  the log.
