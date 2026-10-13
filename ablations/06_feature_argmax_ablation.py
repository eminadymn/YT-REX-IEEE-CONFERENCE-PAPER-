"""
Stage 3 ablation, strongest form: same candidates, same features, no learning.

The regex selector in 04_ranker_ablation.py reads raw text, so a reader can
object that it saw less information than the ranker. This variant removes that
objection: it picks from the *same* candidate set, scored by the *same* feature
vector, and differs from the deployed system in one respect only -- how the
features are combined. The ranker learns a gradient-boosted combination; this
baseline takes the argmax of a single feature.

Which feature is used is selected on the TRAIN split only, by how often its
argmax lands on the positively-labelled candidate. Test predictions are never
consulted during selection.

Inputs (produced by 01_data_preparation.py on Kaggle):
    artifacts/candidates_{key}_{train,test}.parquet
Outputs:
    results/feature_argmax_ablation.csv
"""

import ast
import csv
import os
import re
import sys
import textwrap
from collections import Counter

import pandas as pd

KEYS = ["company", "date", "address", "total"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = os.path.join(ROOT, "artifacts", "candidates_{key}_{split}.parquet")
PREDS = os.path.join(ROOT, "artifacts", "ytrex_predictions.csv")
OUT = os.path.join(ROOT, "results", "feature_argmax_ablation.csv")


def _lift(path, names):
    """Exec named top-level defs / assignments from `path` verbatim."""
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    ns = {"re": re, "Counter": Counter}
    for node in tree.body:
        name = None
        if isinstance(node, ast.FunctionDef):
            name = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in names:
            exec(textwrap.dedent(ast.get_source_segment(src, node)), ns)
    missing = [n for n in names if n not in ns]
    if missing:
        sys.exit(f"ERROR: could not lift {missing} from {path}")
    return ns


_m = _lift(os.path.join(ROOT, "02_model_training.py"),
           ["FEATURES", "normalize_text", "f1_score_text", "em_score"])
FEATURES = _m["FEATURES"]
f1_score_text, em_score = _m["f1_score_text"], _m["em_score"]


def load_gt(path):
    gt = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt[row["base"]] = {k: (row.get(f"gt_{k}") or "") for k in KEYS}
    return gt


def argmax_pick(df, feat, sign):
    """Per image, take the candidate maximising sign*feat. Returns {base: value_str}."""
    s = df.assign(_s=sign * df[feat].astype(float))
    idx = s.groupby("base")["_s"].idxmax()
    return dict(zip(df.loc[idx, "base"], df.loc[idx, "value_str"]))


def train_selection_accuracy(df, feat, sign):
    """How often the argmax lands on a positively-labelled candidate (train only)."""
    s = df.assign(_s=sign * df[feat].astype(float))
    idx = s.groupby("base")["_s"].idxmax()
    return float(df.loc[idx, "label"].astype(int).mean())


def main():
    gt = load_gt(PREDS)
    out_rows, summary = [], {}

    for key in KEYS:
        tr = pd.read_parquet(PARQUET.format(key=key, split="train"))
        te = pd.read_parquet(PARQUET.format(key=key, split="test"))
        feats = [f for f in FEATURES[key] if f in tr.columns]

        # ── choose the single feature on TRAIN only ──
        scored = []
        for f in feats:
            if not pd.api.types.is_numeric_dtype(tr[f]):
                continue
            for sign in (1, -1):
                scored.append((train_selection_accuracy(tr, f, sign), f, sign))
        scored.sort(reverse=True)
        acc, best_f, best_sign = scored[0]
        arrow = "max" if best_sign == 1 else "min"

        # ── apply to TEST ──
        picks = argmax_pick(te, best_f, best_sign)
        bases = sorted(set(te["base"]) & set(gt))
        f1 = [f1_score_text(picks.get(b, ""), gt[b][key]) for b in bases]
        em = [em_score(picks.get(b, ""), gt[b][key]) for b in bases]
        summary[key] = (100 * sum(f1) / len(f1), 100 * sum(em) / len(em))

        print(f"  {key:8s} best single feature: {arrow}({best_f})"
              f"   train sel-acc={100*acc:5.2f}%"
              f"   ->  test F1={summary[key][0]:6.2f}  EM={summary[key][1]:6.2f}")
        print("           runners-up: " + ", ".join(
            f"{'max' if s == 1 else 'min'}({f})={100*a:.1f}%"
            for a, f, s in scored[1:4]))
        out_rows.append({"key": key, "feature": best_f, "direction": arrow,
                         "train_sel_acc": round(100 * acc, 2),
                         "test_f1": round(summary[key][0], 2),
                         "test_em": round(summary[key][1], 2)})

    avg_f1 = sum(v[0] for v in summary.values()) / len(KEYS)
    avg_em = sum(v[1] for v in summary.values()) / len(KEYS)
    print(f"\n  {'AVG':8s} F1={avg_f1:6.2f}  EM={avg_em:6.2f}")

    learnt = {"company": 89.09, "date": 92.12, "address": 83.96,
              "total": 84.15, "AVG": 87.33}
    print("\n  Delta (learnt ranker - single-feature argmax)")
    for k in KEYS:
        print(f"    {k:8s} {learnt[k] - summary[k][0]:+6.2f}")
    print(f"    {'AVG':8s} {learnt['AVG'] - avg_f1:+6.2f}")

    out_rows.append({"key": "AVG", "feature": "-", "direction": "-",
                     "train_sel_acc": "", "test_f1": round(avg_f1, 2),
                     "test_em": round(avg_em, 2)})
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print("\nSaved: results/feature_argmax_ablation.csv")


if __name__ == "__main__":
    main()
