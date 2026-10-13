"""
Stage 3 ablation: learnt ranker vs. keyword/regex selector.

Reviewer request: isolate the contribution of the Small Learnt Ranker by
replacing ONLY Stage 3, holding Stages 1-2 fixed.

Rather than re-running YOLO+TrOCR (which would drift on fp16 / box merging /
max_new_tokens / crop padding), we feed the regex selector the *cached* merged
spatial lines that the deployed rankers consumed -- so Stages 1-2 are bit-identical,
not merely identically configured.

  Stage 1+2  : spatial_cache_test.pkl        (347 images, merged y-sorted lines)
  Stage 3 (a): learnt rankers                -> ytrex_predictions.csv (paper numbers)
  Stage 3 (b): _extract_fields()             -> lifted verbatim from
                                                yolotrocr_rgx_v1_f69_78.py
  Metrics    : normalize_text / f1_score_text / em_score, lifted verbatim from
               02_model_training.py (the functions that produced 87.33 / 65.63)

CPU only, no GPU, no network.
"""

import ast
import csv
import os
import pickle
import re
import sys
import textwrap
from collections import Counter

KEYS = ["company", "date", "address", "total"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "artifacts", "spatial_cache_test.pkl")
PREDS = os.path.join(ROOT, "artifacts", "ytrex_predictions.csv")
REGEX_SRC = os.path.join(ROOT, "baselines", "yolotrocr_rgx_v1_f69_78.py")
TRAIN_SRC = os.path.join(ROOT, "02_model_training.py")
OUT = os.path.join(ROOT, "results", "ranker_ablation_results.csv")


# ── verbatim source lifting ──────────────────────────────────────────────────

def _lift(path, names, class_scope=None):
    """Exec the named top-level (or in-class) function defs from `path` verbatim.

    Guarantees the ablation runs the author's own code, with no re-implementation.
    """
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)

    if class_scope:
        body = next(n.body for n in tree.body
                    if isinstance(n, ast.ClassDef) and n.name == class_scope)
    else:
        body = tree.body

    ns = {"re": re, "Counter": Counter}
    found = set()
    for node in body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(textwrap.dedent(ast.get_source_segment(src, node)), ns)
            found.add(node.name)

    missing = set(names) - found
    if missing:
        sys.exit(f"ERROR: could not lift {missing} from {path}")
    return {n: ns[n] for n in names}


_rgx = _lift(REGEX_SRC, ["_clean_price"])
_rgx.update(_lift(REGEX_SRC, ["_extract_fields"], class_scope="YoloOcrPipeline"))
_met = _lift(TRAIN_SRC, ["normalize_text", "f1_score_text", "em_score"])

# _extract_fields calls the module-level _clean_price; rebind it into that closure
_rgx["_extract_fields"].__globals__["_clean_price"] = _rgx["_clean_price"]

extract_fields = _rgx["_extract_fields"]
f1_score_text = _met["f1_score_text"]
em_score = _met["em_score"]


# ── data ─────────────────────────────────────────────────────────────────────

def load_gt_and_ytrex(path):
    """Ground truth + deployed-ranker predictions, keyed by image base."""
    gt, ytrex = {}, {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            b = row["base"]
            gt[b] = {k: (row.get(f"gt_{k}") or "") for k in KEYS}
            ytrex[b] = {k: (row.get(f"pred_{k}") or "") for k in KEYS}
    return gt, ytrex


def lines_of(record):
    """Reconstruct the regex pipeline's `lines` / `full_text` view of Stage 1-2.

    Mirrors yolotrocr_rgx_v1_f69_78.py:154-155 exactly:
        lines = [t.strip() for t in texts if t.strip()]
        full_text = "\\n".join(lines)
    Cached lines are already sorted top-to-bottom (verified: y1 ascending).
    """
    lines = [l["text"].strip() for l in record["spatial_lines"] if l["text"].strip()]
    return lines, "\n".join(lines)


# ── scoring ──────────────────────────────────────────────────────────────────

def score(preds, gt, bases):
    out = {}
    for k in KEYS:
        f1 = [f1_score_text(preds[b][k], gt[b][k]) for b in bases]
        em = [em_score(preds[b][k], gt[b][k]) for b in bases]
        out[k] = (100 * sum(f1) / len(f1), 100 * sum(em) / len(em))
    out["AVG"] = (sum(v[0] for k, v in out.items()) / len(KEYS),
                  sum(v[1] for k, v in out.items()) / len(KEYS))
    return out


def table(name, s):
    print(f"\n  {name}")
    print(f"  {'KEY':<10}{'F1':>9}{'EM':>9}")
    for k in KEYS + ["AVG"]:
        print(f"  {k.upper():<10}{s[k][0]:>9.2f}{s[k][1]:>9.2f}")


def main():
    cache = pickle.load(open(CACHE, "rb"))
    gt, ytrex = load_gt_and_ytrex(PREDS)

    bases = sorted(set(cache) & set(gt))
    print(f"cache={len(cache)}  preds={len(gt)}  aligned={len(bases)}")
    if len(bases) != len(cache) or len(bases) != len(gt):
        print(f"  WARNING: {len(set(cache) ^ set(gt))} unaligned bases skipped")

    # Stage 3 (b): regex selector on the identical Stage 1-2 output
    regex_preds = {}
    for b in bases:
        lines, full_text = lines_of(cache[b])
        regex_preds[b] = extract_fields(None, full_text, lines)

    s_rgx = score(regex_preds, gt, bases)
    s_ytx = score(ytrex, gt, bases)

    print("\n" + "=" * 46)
    print("  STAGE 3 ABLATION  (Stages 1-2 bit-identical)")
    print("=" * 46)
    table("Stage 3 = keyword/regex selector", s_rgx)
    table("Stage 3 = learnt rankers (deployed)", s_ytx)

    print("\n  Delta (learnt - regex)")
    print(f"  {'KEY':<10}{'dF1':>9}{'dEM':>9}")
    for k in KEYS + ["AVG"]:
        print(f"  {k.upper():<10}{s_ytx[k][0]-s_rgx[k][0]:>+9.2f}"
              f"{s_ytx[k][1]-s_rgx[k][1]:>+9.2f}")

    # sanity: do we reproduce the paper's numbers from the same CSV?
    with open(PREDS, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)]
    print("\n  Sanity check vs. paper (Table II):")
    for k in KEYS:
        col = 100 * sum(float(r[f"{k}_f1"]) for r in rows) / len(rows)
        print(f"    {k:<8} recomputed F1 {s_ytx[k][0]:6.2f} | "
              f"csv column {col:6.2f}")
    avg_col = sum(100 * sum(float(r[f"{k}_f1"]) for r in rows) / len(rows)
                  for k in KEYS) / len(KEYS)
    print(f"    {'AVG':<8} recomputed F1 {s_ytx['AVG'][0]:6.2f} | "
          f"csv column {avg_col:6.2f} | paper 87.33")

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage3", "key", "f1", "em"])
        for name, s in (("regex", s_rgx), ("learnt", s_ytx)):
            for k in KEYS + ["AVG"]:
                w.writerow([name, k, f"{s[k][0]:.2f}", f"{s[k][1]:.2f}"])
    print("\nSaved: results/ranker_ablation_results.csv")


if __name__ == "__main__":
    main()
