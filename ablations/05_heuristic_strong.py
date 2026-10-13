"""
Stage 3 ablation, strengthened baseline ("heuristic v2").

Purpose: pre-empt a straw-man objection to 04_ranker_ablation.py. The v1 regex
selector loses `total` largely to two fixable defects -- `total` has no word
boundary so it fires inside SUBTOTAL (34 cases), and re.search takes the FIRST
keyword occurrence while `[^\\d]*` bridges newlines, so it grabs a digit from an
unrelated line (171 cases).

v2 is what a competent engineer writes by hand, still with NO learning:
  total   : total-keyword lines with subtotal/qty/discount/round/gst excluded,
            take the LAST anchor, read the price on that line (or the next two),
            fall back to the page max price.
  company : receipt headers are printed larger -- take the tallest non-numeric,
            non-address line in the top 30% of the page (box height from Stage 1).
  address : take the longest CONTIGUOUS run of address-like lines in the top 60%,
            instead of joining four scattered matches from anywhere on the page.
  date    : unchanged from v1 (already 91.55 F1, at the OCR ceiling).

Stages 1-2 stay bit-identical (same cached merged lines). CPU only.
"""

import csv
import importlib.util
import os
import pickle
import re

KEYS = ["company", "date", "address", "total"]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


abl = _load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "04_ranker_ablation.py"), "abl")
OUT = os.path.join(abl.ROOT, "results", "ranker_ablation_v2_results.csv")

# same price grammar as the paper's Stage 3 candidate generator (Sec. III-C)
PRICE = re.compile(r'(?:RM\s*)?(\d{1,6}(?:[,.]\d{3})*[,.]\d{2})')

TOTAL_POS = re.compile(
    r'\b(?:grand\s*total|nett?\s*total|total\s*amount|amount\s*due|'
    r'total\s*due|total\s*payable|total)\b', re.I)
TOTAL_NEG = re.compile(
    r'sub\s*-?\s*total|subtotal|total\s*(?:item|qty|quantity|discount|saving|'
    r'excl|exclu|before)|round|gst|tax\s*total|change', re.I)

ADDR_KW = re.compile(
    r'\b(jalan|jln|lot|no\.|block|blk|floor|level|road|street|avenue|highway|'
    r'taman|bandar|off|km\b|kompleks|plaza|tower|building|\d{5})\b', re.I)
NOT_COMPANY = re.compile(
    r'\b(jalan|jln|lot|no\.|blk|floor|level|tel|phone|fax|www|http|gst|'
    r'invoice|receipt|roc)\b', re.I)
NUMERIC_ONLY = re.compile(r'^[\d\s\.\-\:\/\(\)\*]+$')


def _price_floats(text):
    out = []
    for m in PRICE.finditer(text):
        raw = m.group(1)
        norm = re.sub(r'[,.](?=\d{3}\b)', '', raw).replace(',', '.')
        try:
            out.append((float(norm), norm))
        except ValueError:
            pass
    return out


def extract_v2(lines):
    """lines: list of cached Stage 1-2 dicts (text + geometry)."""
    texts = [l["text"].strip() for l in lines]
    full_text = "\n".join(t for t in texts if t)

    # ── TOTAL ──
    total = ""
    anchors = [i for i, t in enumerate(texts)
               if TOTAL_POS.search(t) and not TOTAL_NEG.search(t)]
    for i in reversed(anchors):
        cand = _price_floats(texts[i])
        if not cand:
            for j in (i + 1, i + 2):
                if j < len(texts):
                    cand = _price_floats(texts[j])
                    if cand:
                        break
        if cand:
            total = cand[-1][1]
            break
    if not total:
        allp = _price_floats(full_text)
        if allp:
            total = max(allp, key=lambda x: x[0])[1]

    # ── DATE (v1, unchanged) ──
    date = ""
    for pat in [
        r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b',
        r'\b(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b',
        r'\b(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)'
        r'[a-z]*[\s,]+\d{2,4})\b',
    ]:
        m = re.search(pat, full_text, re.IGNORECASE)
        if m:
            date = m.group(1)
            break

    # ── COMPANY: tallest plausible header line in the top 30% ──
    company = ""
    head = [l for l in lines
            if l["rel_y"] <= 0.30 and l["text"].strip()
            and not NUMERIC_ONLY.match(l["text"].strip())
            and not NOT_COMPANY.search(l["text"])
            and len(l["text"].strip()) >= 3]
    if head:
        company = max(head, key=lambda l: l["h_ratio"])["text"].strip()
    else:
        for t in texts[:8]:
            if t and not NUMERIC_ONLY.match(t) and not NOT_COMPANY.search(t):
                company = t
                break

    # ── ADDRESS: longest contiguous address-like run in the top 60% ──
    flags = [bool(ADDR_KW.search(l["text"])) and l["rel_y"] <= 0.60
             for l in lines]
    best, cur = [], []
    for i, f in enumerate(flags):
        if f:
            cur.append(i)
        else:
            if len(cur) > len(best):
                best = cur
            cur = []
    if len(cur) > len(best):
        best = cur
    if best:
        address = " ".join(texts[i] for i in best[:4])
    else:
        address = " ".join(t for l, t in zip(lines, texts)
                           if ADDR_KW.search(t))[:200]

    return {"company": company, "date": date, "address": address, "total": total}


def main():
    cache = pickle.load(open(abl.CACHE, "rb"))
    gt, ytrex = abl.load_gt_and_ytrex(abl.PREDS)
    bases = sorted(set(cache) & set(gt))

    v1, v2 = {}, {}
    for b in bases:
        lines, full_text = abl.lines_of(cache[b])
        v1[b] = abl.extract_fields(None, full_text, lines)
        v2[b] = extract_v2(cache[b]["spatial_lines"])

    rows = [("regex v1 (author's)", abl.score(v1, gt, bases)),
            ("heuristic v2 (strengthened)", abl.score(v2, gt, bases)),
            ("learnt rankers (deployed)", abl.score(ytrex, gt, bases))]

    print("=" * 62)
    print("  STAGE 3 ABLATION -- Stages 1-2 bit-identical, N=%d" % len(bases))
    print("=" * 62)
    print(f"\n  {'Stage 3':<30}" + "".join(f"{k[:7].upper():>9}" for k in KEYS)
          + f"{'AVG':>9}")
    for name, s in rows:
        print(f"  {name:<30}" + "".join(f"{s[k][0]:>9.2f}" for k in KEYS)
              + f"{s['AVG'][0]:>9.2f}")
    print("\n  (EM)")
    for name, s in rows:
        print(f"  {name:<30}" + "".join(f"{s[k][1]:>9.2f}" for k in KEYS)
              + f"{s['AVG'][1]:>9.2f}")

    sl, s2 = rows[2][1], rows[1][1]
    print("\n  Delta (learnt - heuristic v2)")
    print(f"  {'':<30}" + "".join(f"{sl[k][0]-s2[k][0]:>+9.2f}" for k in KEYS)
          + f"{sl['AVG'][0]-s2['AVG'][0]:>+9.2f}")

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage3", "key", "f1", "em"])
        for name, s in rows:
            for k in KEYS + ["AVG"]:
                w.writerow([name, k, f"{s[k][0]:.2f}", f"{s[k][1]:.2f}"])
    print("\nSaved: results/ranker_ablation_v2_results.csv")


if __name__ == "__main__":
    main()
