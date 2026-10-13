"""
YT-Rex Stage 2 — SMALL LEARNT RANKER (per-key)
==============================================
Train a pointwise Small Learnt Ranker for EACH of the 4 keys
(company, date, address, total) on SROIE train/, evaluate on test/.
- 5-fold GroupKFold CV (group by image base) for every key
- Refit on full train, predict on test
- Report F1 / EM / ANLS per key
- Primary metric: ANLS (Document AI standard)

Paper mapping: this is the learning half of Stage 3 (Sec. III-C) and produces
Table II. The paper reports F1 and EM; ANLS is logged here but not published.
"Stage 2" in this script's banners counts scripts, not paper stages.
"""

import os, json, re, pickle, time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier

# ── Paths ───────────────────────────────────────────────────────────────────
# Defaults are the Kaggle mounts of the paper run; override via environment.
WORK_DIR     = os.environ.get("YTREX_WORK_DIR", "/kaggle/working")
DATA_ROOT    = os.environ.get(
    "YTREX_DATA_ROOT",
    "/kaggle/input/datasets/urbikn/sroie-datasetv2/SROIE2019")
TRAIN_GT_DIR = f"{DATA_ROOT}/train/entities"
TEST_GT_DIR  = f"{DATA_ROOT}/test/entities"
KEYS         = ["company", "date", "address", "total"]


# ── Feature sets per key (must match 01_data_preparation.py) ────────────────
FEATURES = {
    "total": [
        "rel_y", "x2_ratio", "w_ratio", "h_ratio", "rel_x",
        "has_tier1_keyword", "has_tier2_keyword", "has_tier1_in_ctx",
        "has_cash_keyword", "has_cash_in_ctx",
        "price_string_pos_ratio",
        "is_largest_in_row", "is_global_max", "ratio_to_global_max",
        "n_prices_in_line", "line_idx_norm", "is_bottom_quarter", "log_val",
        "dist_to_tier1_above", "dist_to_tier2_above", "dist_to_cash_above",
        "is_below_tier1", "is_below_tier2", "is_below_cash", "n_lines_total",
        # iter2
        "dist_to_tier1_below", "dist_to_tier2_below",
        "is_above_tier1", "is_above_tier2",
        "n_tier1_in_image", "n_tier2_in_image",
        "val_is_round", "val_rank", "val_rank_norm", "is_top2_val",
        "frac_below_global_max",
    ],
    "company": [
        "is_merged", "rel_y", "rel_x", "w_ratio", "h_ratio", "x2_ratio",
        "line_idx_norm", "is_top_15", "is_top_25", "is_top_35",
        "n_words", "char_len", "upper_ratio", "n_digits", "digit_ratio",
        "has_comp_kw", "has_skip_kw", "has_phone", "has_comp_kw_below",
        "n_lines_total",
    ],
    "date": [
        "rel_y", "rel_x", "w_ratio", "h_ratio", "line_idx_norm",
        "is_top_half", "is_top_25", "is_top_50",
        "pat_id", "pat_is_dmy4", "pat_is_ymd", "pat_is_dmy2", "pat_is_text",
        "n_dates_image", "match_start", "match_len", "match_pos_ratio",
        "has_date_kw_line", "dist_to_kw_above", "dist_to_kw_below",
        "kw_within_2_above", "n_lines_total",
    ],
    "address": [
        "start_idx", "n_lines",
        "start_rel_y", "end_rel_y", "dy_span",
        "avg_w_ratio", "max_w_ratio", "avg_n_words", "total_chars",
        "avg_chars_per_line",
        "start_idx_norm",
        "is_top_15", "is_top_30", "is_top_50",
        "addr_hits", "frac_addr", "post_hits", "has_postcode",
        "frac_digit_lines", "ends_at_postcode", "n_lines_total",
        # iter2
        "state_hits", "has_state", "block_hits", "has_block",
        "first_line_is_block",
        "comma_count", "comma_per_line",
        "ends_at_state", "has_addr_and_post", "has_addr_and_state",
    ],
}


# ── Metrics ─────────────────────────────────────────────────────────────────
def normalize_text(t):
    if not isinstance(t, str): t = str(t)
    t = t.lower().strip()
    t = re.sub(r'^rm\s*', '', t)
    t = t.replace(',', '.')
    t = re.sub(r'[^\w\s\.]', '', t)
    return re.sub(r'\s+', ' ', t).strip()


def f1_score_text(pred, truth):
    p, t = normalize_text(pred).split(), normalize_text(truth).split()
    common = Counter(p) & Counter(t); same = sum(common.values())
    if not p or not t: return float(p == t)
    if same == 0: return 0.0
    pr = same / len(p); rc = same / len(t)
    return (2 * pr * rc) / (pr + rc)


def em_score(pred, truth):
    return int(normalize_text(pred) == normalize_text(truth))


def _levenshtein(a, b):
    if a == b: return 0
    if not a:  return len(b)
    if not b:  return len(a)
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j - 1] + 1,    # ins
                         prev[j]    + 1,    # del
                         prev[j - 1] + cost)  # sub
        prev = cur
    return prev[-1]


def anls_score(pred, truth, tau=0.5):
    """Average Normalized Levenshtein Similarity (DocVQA-style).
       Threshold tau: if NL similarity < tau, score = 0.
    """
    p, t = normalize_text(pred), normalize_text(truth)
    if not p and not t:
        return 1.0
    if not p or not t:
        return 0.0
    d = _levenshtein(p, t)
    nls = 1.0 - d / max(len(p), len(t))
    return nls if nls >= tau else 0.0


# ── GT loader ───────────────────────────────────────────────────────────────
def load_gt(path):
    default = {k: "" for k in KEYS}
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    try:
        return {k: json.loads(content).get(k, "") for k in KEYS}
    except json.JSONDecodeError:
        pass
    out = dict(default)
    for line in content.splitlines():
        for sep in (":", "="):
            if sep in line:
                k, v = line.split(sep, 1); k = k.strip().lower()
                if k in out: out[k] = v.strip()
                break
    return out


# ── Small Learnt Ranker (pointwise) ─────────────────────────────────────────
class SmallLearntRanker:
    """Pointwise ranker over per-image candidates (HGB classifier).
       Per-group argmax of P(label=1) → final pick."""
    def __init__(self, **kw):
        self.model = Pipeline([
            ("clf", HistGradientBoostingClassifier(
                max_depth=kw.get("max_depth", 4),
                max_iter=kw.get("max_iter", 400),
                learning_rate=kw.get("lr", 0.05),
                l2_regularization=kw.get("l2", 2.0),
                min_samples_leaf=kw.get("min_leaf", 30),
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=20,
                class_weight="balanced", random_state=42)),
        ])

    def fit(self, X, y):
        self.model.fit(X, y); return self

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]

    def pick_per_group(self, df, X):
        scores = self.score(X)
        df = df.assign(_p=scores)
        out = {}
        for base, grp in df.groupby("base"):
            out[base] = grp.loc[grp["_p"].idxmax(), "value_str"]
        return out


# ── CV: token-F1 on val/train split using ranker picks ──────────────────────
def f1_on_split(df_split, ranker, X, gt_dir, key):
    picks = ranker.pick_per_group(df_split, X)
    f1s = []
    bases = set(df_split["base"].unique())
    for base in bases:
        gt = load_gt(os.path.join(gt_dir, base + ".txt"))
        f1s.append(f1_score_text(picks.get(base, ""), gt.get(key, "")))
    return float(np.mean(f1s)) if f1s else 0.0


def cv_one_key(key, df_train, n_splits=5):
    feats = FEATURES[key]
    X = df_train[feats].values
    y = df_train["label"].values
    groups = df_train["base"].values
    gkf = GroupKFold(n_splits=n_splits)
    fold_tr, fold_va = [], []
    print(f"\n  ── 5-Fold CV : {key.upper()} ──")
    print(f"    rows={len(df_train)}  positives={int(y.sum())}  "
          f"images={df_train['base'].nunique()}  feats={len(feats)}")
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), 1):
        tr = df_train.iloc[tr_idx]; va = df_train.iloc[va_idx]
        # Edge case: some folds may have very few positives — skip CV-internal
        # score, still fit so we report something.
        r = SmallLearntRanker().fit(tr[feats].values, tr["label"].values)
        f1_tr = f1_on_split(tr, r, tr[feats].values, TRAIN_GT_DIR, key)
        f1_va = f1_on_split(va, r, va[feats].values, TRAIN_GT_DIR, key)
        fold_tr.append(f1_tr); fold_va.append(f1_va)
        print(f"    Fold {fold}: train_F1={f1_tr*100:6.2f}%  val_F1={f1_va*100:6.2f}%  "
              f"(n_train_imgs={tr['base'].nunique()}, "
              f"n_val_imgs={va['base'].nunique()})")
    print(f"    CV mean: train={np.mean(fold_tr)*100:.2f}% ± {np.std(fold_tr)*100:.2f}%  "
          f"|  val={np.mean(fold_va)*100:.2f}% ± {np.std(fold_va)*100:.2f}%")


def fit_full_and_predict(key, df_train, df_test):
    feats = FEATURES[key]
    final = SmallLearntRanker().fit(df_train[feats].values,
                                    df_train["label"].values)
    try:
        from sklearn.inspection import permutation_importance
        pi = permutation_importance(
            final.model, df_train[feats].values, df_train["label"].values,
            n_repeats=3, random_state=42, n_jobs=-1)
        top = sorted(zip(feats, pi.importances_mean), key=lambda kv: -kv[1])[:8]
        print(f"    [{key}] top-8 perm importance: " +
              ", ".join(f"{n}={w:+.3f}" for n, w in top))
    except Exception as e:
        print(f"    perm-importance skipped: {e}")
    picks = final.pick_per_group(df_test, df_test[feats].values)
    return final, picks


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 64)
    print("  STAGE 2 — PER-KEY SMALL LEARNT RANKER  |  5-Fold CV → Test")
    print("=" * 64)

    # Load candidate parquets for all keys
    df_train, df_test = {}, {}
    for k in KEYS:
        df_train[k] = pd.read_parquet(
            os.path.join(WORK_DIR, f"candidates_{k}_train.parquet"))
        df_test[k]  = pd.read_parquet(
            os.path.join(WORK_DIR, f"candidates_{k}_test.parquet"))
        print(f"  [{k}] train rows={len(df_train[k])}  pos={int(df_train[k]['label'].sum())} "
              f"| test rows={len(df_test[k])}  pos={int(df_test[k]['label'].sum())}")

    with open(os.path.join(WORK_DIR, "split_meta.json")) as f:
        split = json.load(f)
    test_bases = split["test_bases"]
    try:
        with open(os.path.join(WORK_DIR, "test_timings.json")) as f:
            test_timings = json.load(f)
    except FileNotFoundError:
        test_timings = {}

    # ── 5-Fold CV per key ───────────────────────────────────────────────────
    for k in KEYS:
        cv_one_key(k, df_train[k], n_splits=5)

    # ── Refit & predict on test ─────────────────────────────────────────────
    print("\n  ── Refit on full train + predict on TEST ──")
    rankers, all_picks = {}, {}
    per_key_extract_time = {k: 0.0 for k in KEYS}
    for k in KEYS:
        t0 = time.perf_counter()
        rankers[k], all_picks[k] = fit_full_and_predict(k, df_train[k], df_test[k])
        per_key_extract_time[k] = time.perf_counter() - t0
        with open(os.path.join(WORK_DIR, f"ranker_{k}.pkl"), "wb") as f:
            pickle.dump({"model": rankers[k].model, "features": FEATURES[k]}, f)

    # ── Final evaluation on test ────────────────────────────────────────────
    rows, pred_rows = [], []
    for base in test_bases:
        gt = load_gt(os.path.join(TEST_GT_DIR, base + ".txt"))
        pred = {k: all_picks[k].get(base, "") for k in KEYS}
        row = {"base": base}
        for k in KEYS:
            row[f"{k}_f1"]   = f1_score_text(pred[k], gt.get(k, ""))
            row[f"{k}_em"]   = em_score(pred[k], gt.get(k, ""))
            row[f"{k}_anls"] = anls_score(pred[k], gt.get(k, ""), tau=0.5)
        rows.append(row)
        pred_rows.append({
            "base": base,
            **{f"pred_{k}": pred[k] for k in KEYS},
            **{f"gt_{k}":   gt.get(k, "") for k in KEYS},
            **{f"{k}_f1":   row[f"{k}_f1"]   for k in KEYS},
            **{f"{k}_em":   row[f"{k}_em"]   for k in KEYS},
            **{f"{k}_anls": row[f"{k}_anls"] for k in KEYS},
        })
    pd.DataFrame(pred_rows).to_csv(
        os.path.join(WORK_DIR, "ytrex_predictions.csv"), index=False)
    df_res = pd.DataFrame(rows)
    n_imgs = len(df_res)
    avg_ocr_ms = (1000.0 * np.mean([test_timings.get(b, 0.0) for b in test_bases])
                  if test_timings else float("nan"))

    # Per-image extract time = total fit+predict time / n_imgs (rough; mostly fit dominates,
    # but predict is what test-time costs — re-compute predict-only timing for fairness).
    predict_only_ms = {}
    for k in KEYS:
        feats = FEATURES[k]
        t0 = time.perf_counter()
        _ = rankers[k].pick_per_group(df_test[k], df_test[k][feats].values)
        predict_only_ms[k] = 1000.0 * (time.perf_counter() - t0) / max(n_imgs, 1)

    print("\n" + "=" * 86)
    print(f"  TEST RESULTS  (N={n_imgs})   primary metric: ANLS")
    print("=" * 86)
    print(f"  {'KEY':<10}{'F1':>9}{'EM':>9}{'ANLS':>9}"
          f"{'EXTRACT_ms':>14}{'TOTAL_ms/img':>16}")
    print("  " + "-" * 80)
    f1s, ems, anlss = [], [], []
    for k in KEYS:
        f1   = df_res[f"{k}_f1"].mean()   * 100
        em   = df_res[f"{k}_em"].mean()   * 100
        anls = df_res[f"{k}_anls"].mean() * 100
        f1s.append(f1); ems.append(em); anlss.append(anls)
        extract_ms = predict_only_ms[k]
        total_ms   = avg_ocr_ms + extract_ms
        print(f"  {k.upper():<10}{f1:8.2f}%{em:8.2f}%{anls:8.2f}%"
              f"{extract_ms:12.3f}  {total_ms:14.2f}")
    print("  " + "-" * 80)
    print(f"  {'AVERAGE':<10}{sum(f1s)/len(f1s):8.2f}%"
          f"{sum(ems)/len(ems):8.2f}%"
          f"{sum(anlss)/len(anlss):8.2f}%")
    print(f"\n  Shared OCR pipeline (YOLO+Merge+TrOCR) avg/image: {avg_ocr_ms:.2f} ms")
    print(f"  (TOTAL_ms/img = shared OCR + per-key extraction)")
    print("=" * 86)

    df_res.to_csv(os.path.join(WORK_DIR, "ytrex_ranker_test_results.csv"),
                  index=False)
    print("  Saved: ranker_{company,date,address,total}.pkl, "
          "ytrex_predictions.csv, ytrex_ranker_test_results.csv")


if __name__ == "__main__":
    main()
