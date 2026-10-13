"""
YT-Rex Stage 3 — ERROR TAXONOMY (per-key, attribution)
======================================================
For every (key, image) in the test split where ANLS < 1.0, classify the
error into one of three pipeline-stage buckets:

  YOLO_Miss        — GT not present in any OCR line  → detection failed
  TrOCR_Error      — partial GT presence, recognition noisy
  Wrong_Selection  — GT effectively in OCR stream but ranker picked wrong

Unified rule using token-F1 between GT and the BEST OCR line/candidate:
  YOLO_Miss        : S < 0.30
  TrOCR_Error      : 0.30 <= S < 0.85
  Wrong_Selection  : S >= 0.85  AND  pred != GT

For TOTAL, we additionally check the digit sequence (paper definition):
  - YOLO_Miss if GT digit string absent from every OCR line.

Paper mapping: this is the error classification of Sec. IV-F (Table IV). A
prediction counts as correct when its ANLS against the GT is >= 0.90.
"Stage 3" in this script's banners counts scripts, not paper stages.

Output:
  <YTREX_WORK_DIR>/error_taxonomy.csv  (per-key counts + percentages)
  Printed 4×3 table to stdout.
"""

import os, json, re, pickle
from collections import Counter
import pandas as pd

# Defaults are the Kaggle mounts of the paper run; override via environment.
WORK_DIR     = os.environ.get("YTREX_WORK_DIR", "/kaggle/working")
DATA_ROOT    = os.environ.get(
    "YTREX_DATA_ROOT",
    "/kaggle/input/datasets/urbikn/sroie-datasetv2/SROIE2019")
TEST_GT_DIR  = f"{DATA_ROOT}/test/entities"
KEYS         = ["company", "date", "address", "total"]


# ── Text normalization & token-F1 ───────────────────────────────────────────
def normalize_text(t):
    if not isinstance(t, str): t = str(t)
    t = t.lower().strip()
    t = re.sub(r'^rm\s*', '', t)
    t = t.replace(',', '.')
    t = re.sub(r'[^\w\s\.]', '', t)
    return re.sub(r'\s+', ' ', t).strip()


def token_f1(pred, truth):
    p = normalize_text(pred).split()
    t = normalize_text(truth).split()
    common = Counter(p) & Counter(t); same = sum(common.values())
    if not p or not t: return float(p == t)
    if same == 0: return 0.0
    pr = same / len(p); rc = same / len(t)
    return (2 * pr * rc) / (pr + rc)


def load_gt(path):
    default = {k: "" for k in KEYS}
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    try:
        data = json.loads(content)
        return {k: data.get(k, "") for k in KEYS}
    except json.JSONDecodeError:
        pass
    result = dict(default)
    for line in content.splitlines():
        for sep in (":", "="):
            if sep in line:
                k, v = line.split(sep, 1)
                k = k.strip().lower()
                if k in result:
                    result[k] = v.strip()
                break
    return result


# ── Best-S over OCR lines for a given key ───────────────────────────────────
def best_match_score(gt_text, spatial_lines, key):
    if not gt_text or not spatial_lines:
        return 0.0, ""
    # For TOTAL: try every line and pick max token-F1 (gt is short, ~1 line)
    # For COMPANY/DATE: same
    # For ADDRESS: gt spans multiple lines; also try sliding window 1..7
    best_s, best_text = 0.0, ""
    n = len(spatial_lines)
    if key == "address":
        # sliding window joins
        for i in range(min(30, n)):
            for L in range(1, 8):
                j = i + L
                if j > n: break
                joined = ", ".join(s["text"].strip() for s in spatial_lines[i:j])
                s = token_f1(joined, gt_text)
                if s > best_s:
                    best_s, best_text = s, joined
    else:
        for sl in spatial_lines:
            s = token_f1(sl["text"], gt_text)
            if s > best_s:
                best_s, best_text = s, sl["text"]
    return best_s, best_text


# ── Digit-level check for TOTAL (paper definition) ──────────────────────────
def total_digits_in_ocr(gt_total, spatial_lines):
    digits = re.sub(r'[^\d]', '', gt_total)
    if not digits:
        return False
    return any(digits in re.sub(r'[^\d]', '', sl["text"])
               for sl in spatial_lines)


# ── Categorize one (key, image) ─────────────────────────────────────────────
ANLS_CORRECT_THR = 0.9


def categorize(key, pred, gt, spatial_lines, anls):
    if not gt:
        return "no_gt"
    # ANLS-based correctness: aligned with paper's primary metric.
    if anls >= ANLS_CORRECT_THR:
        return "correct"

    best_s, _ = best_match_score(gt, spatial_lines, key)

    if key == "total":
        # Stronger digit-level check trumps token-F1 thresholds
        if not total_digits_in_ocr(gt, spatial_lines):
            return "YOLO_Miss"
        # GT digits present in OCR
        if best_s >= 0.85:
            return "Wrong_Selection"
        return "TrOCR_Error"

    # company / date / address
    if best_s < 0.30:
        return "YOLO_Miss"
    if best_s < 0.85:
        return "TrOCR_Error"
    return "Wrong_Selection"


def main():
    print("=" * 64)
    print("  STAGE 3 — ERROR TAXONOMY (per-key, 4×3 table)")
    print("=" * 64)

    with open(os.path.join(WORK_DIR, "spatial_cache_test.pkl"), "rb") as f:
        cache = pickle.load(f)
    # IMPORTANT: read as strings to preserve "193.00" → not coerced to 193.0
    preds = pd.read_csv(os.path.join(WORK_DIR, "ytrex_predictions.csv"),
                        dtype=str, keep_default_na=False)

    cats = ["YOLO_Miss", "TrOCR_Error", "Wrong_Selection"]
    counts = {k: {c: 0 for c in cats} for k in KEYS}
    correct_counts = {k: 0 for k in KEYS}
    no_gt = {k: 0 for k in KEYS}
    err_examples = {k: [] for k in KEYS}

    for _, row in preds.iterrows():
        base = row["base"]
        lines = cache.get(base, {}).get("spatial_lines", [])
        gt_all = load_gt(os.path.join(TEST_GT_DIR, base + ".txt"))
        for k in KEYS:
            pred = row.get(f"pred_{k}", "") or ""
            gt   = gt_all.get(k, "") or ""
            try:
                anls = float(row.get(f"{k}_anls", 0.0))
            except (TypeError, ValueError):
                anls = 0.0
            cat  = categorize(k, str(pred), str(gt), lines, anls)
            if cat == "correct":
                correct_counts[k] += 1
            elif cat == "no_gt":
                no_gt[k] += 1
            else:
                counts[k][cat] += 1
                if len(err_examples[k]) < 3:
                    err_examples[k].append((base, cat, str(pred)[:60], str(gt)[:60]))

    N = len(preds)
    print(f"\n  Total test images: {N}")
    print(f"\n  {'KEY':<10}{'YOLO_Miss':>12}{'TrOCR_Err':>12}{'Wrong_Sel':>12}"
          f"{'Errors':>10}{'Correct':>10}{'NoGT':>8}")
    print("  " + "-" * 74)
    rows_out = []
    for k in KEYS:
        ym = counts[k]["YOLO_Miss"]
        te = counts[k]["TrOCR_Error"]
        ws = counts[k]["Wrong_Selection"]
        err = ym + te + ws
        cor = correct_counts[k]
        ng  = no_gt[k]
        print(f"  {k.upper():<10}{ym:>12}{te:>12}{ws:>12}{err:>10}{cor:>10}{ng:>8}")
        rows_out.append({
            "key": k, "YOLO_Miss": ym, "TrOCR_Error": te,
            "Wrong_Selection": ws, "errors_total": err,
            "correct": cor, "no_gt": ng, "N": N,
        })
    print("  " + "-" * 74)
    print("\n  % of errors per key:")
    print(f"  {'KEY':<10}{'YOLO_Miss%':>12}{'TrOCR_Err%':>12}{'Wrong_Sel%':>12}")
    print("  " + "-" * 50)
    for k in KEYS:
        ym = counts[k]["YOLO_Miss"]; te = counts[k]["TrOCR_Error"]; ws = counts[k]["Wrong_Selection"]
        err = max(ym + te + ws, 1)
        print(f"  {k.upper():<10}"
              f"{100*ym/err:>11.1f}%{100*te/err:>11.1f}%{100*ws/err:>11.1f}%")

    print("\n  Example errors (up to 3 per key):")
    for k in KEYS:
        print(f"  [{k.upper()}]")
        for base, cat, pred, gt in err_examples[k]:
            print(f"    {base[-12:]} {cat:<16} pred='{pred}' gt='{gt}'")

    pd.DataFrame(rows_out).to_csv(
        os.path.join(WORK_DIR, "error_taxonomy.csv"), index=False)
    print("\n  Saved: error_taxonomy.csv")
    print("=" * 64)


if __name__ == "__main__":
    main()
