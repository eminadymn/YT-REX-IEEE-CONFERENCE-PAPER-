"""
YT-Rex Stage 1 — DATA PREPARATION (train/ + test/ separate)
===========================================================
1) YOLO detection
2) Horizontal Box Merging (Selective Search)
3) TrOCR (fp16, batched)
4) Per-key CANDIDATE extraction:
     - total    : every price match (existing)
     - company  : first ~15 lines as singletons + 2-line merges
     - date     : every regex date match in any line
     - address  : contiguous line-blocks (length 1..7)
5) Per-image timing (YOLO+Merge+TrOCR) for test split

Paper mapping: steps 1-3 are Stage 1 (detection) and Stage 2 (recognition);
step 4 is the candidate generation + feature extraction half of Stage 3
(Sec. III). "Stage 1" in this script's banners counts scripts, not paper stages.

Outputs (in YTREX_WORK_DIR, default /kaggle/working/):
  - spatial_cache_{train,test}.pkl
  - candidates_total_{train,test}.parquet
  - candidates_company_{train,test}.parquet
  - candidates_date_{train,test}.parquet
  - candidates_address_{train,test}.parquet
  - test_timings.json
  - split_meta.json
"""

import os, json, time, re, pickle, gc, subprocess
from collections import Counter

import numpy as np
import pandas as pd

try:
    import ultralytics  # noqa
except ImportError:
    subprocess.run(["pip", "install", "-q", "ultralytics"], check=True)

import torch
from PIL import Image, ImageFilter, ImageOps
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ── Paths / constants ───────────────────────────────────────────────────────
# Defaults are the Kaggle mounts of the paper run; override via environment.
YOLO_PATH      = os.environ.get(
    "YTREX_YOLO_PATH",
    "/kaggle/input/models/muhammedeminadyaman/yolov8/pytorch/default/1/best.pt")
DATA_ROOT      = os.environ.get(
    "YTREX_DATA_ROOT",
    "/kaggle/input/datasets/urbikn/sroie-datasetv2/SROIE2019")
TRAIN_IMG_DIR  = f"{DATA_ROOT}/train/img"
TRAIN_GT_DIR   = f"{DATA_ROOT}/train/entities"
TEST_IMG_DIR   = f"{DATA_ROOT}/test/img"
TEST_GT_DIR    = f"{DATA_ROOT}/test/entities"
OUT_DIR        = os.environ.get("YTREX_WORK_DIR", "/kaggle/working")
KEYS           = ["company", "date", "address", "total"]

os.makedirs(OUT_DIR, exist_ok=True)


# ── GT loader ───────────────────────────────────────────────────────────────
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


# ── Text normalization & token-F1 (for candidate labeling) ──────────────────
def _norm(t):
    if not isinstance(t, str): t = str(t)
    t = t.lower().strip()
    t = re.sub(r'^rm\s*', '', t)
    t = t.replace(',', '.')
    t = re.sub(r'[^\w\s\.]', '', t)
    return re.sub(r'\s+', ' ', t).strip()


def _token_f1(pred, truth):
    p, t = _norm(pred).split(), _norm(truth).split()
    common = Counter(p) & Counter(t); same = sum(common.values())
    if not p or not t: return float(p == t)
    if same == 0: return 0.0
    pr = same / len(p); rc = same / len(t)
    return (2 * pr * rc) / (pr + rc)


# ── Horizontal Box Merging (Selective Search) ───────────────────────────────
class HorizontalBoxMerger:
    def __init__(self, y_overlap_thr=0.5, x_gap_ratio=0.04):
        self.y_overlap_thr = y_overlap_thr
        self.x_gap_ratio  = x_gap_ratio

    @staticmethod
    def _y_overlap(a, b):
        ay1, ay2 = a[1], a[3]
        by1, by2 = b[1], b[3]
        inter = max(0, min(ay2, by2) - max(ay1, by1))
        denom = min(ay2 - ay1, by2 - by1)
        return inter / denom if denom > 0 else 0.0

    def merge(self, boxes, img_w):
        if len(boxes) == 0:
            return []
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        rows = []
        for b in boxes:
            placed = False
            for row in rows:
                if self._y_overlap(b, row[-1]) >= self.y_overlap_thr:
                    row.append(b)
                    placed = True
                    break
            if not placed:
                rows.append([b])
        gap_thr = self.x_gap_ratio * img_w
        merged = []
        for row in rows:
            row.sort(key=lambda b: b[0])
            cur = list(row[0])
            for nxt in row[1:]:
                gap = nxt[0] - cur[2]
                if gap <= gap_thr:
                    cur[0] = min(cur[0], nxt[0])
                    cur[1] = min(cur[1], nxt[1])
                    cur[2] = max(cur[2], nxt[2])
                    cur[3] = max(cur[3], nxt[3])
                else:
                    merged.append(tuple(cur))
                    cur = list(nxt)
            merged.append(tuple(cur))
        merged.sort(key=lambda b: (b[1], b[0]))
        return merged


# ── YOLO + TrOCR runner (fp16, global batched) ──────────────────────────────
class FastYoloTrOCRRunner:
    def __init__(self, yolo_path, batch_size=128, pad_x=3, pad_y=2,
                 use_fp16=True, max_new_tokens=32):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fp16 = use_fp16 and self.device == "cuda"
        self.max_new_tokens = max_new_tokens
        self.yolo = YOLO(yolo_path)
        if self.device == "cuda":
            self.yolo.to("cuda")
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
        dtype = torch.float16 if self.use_fp16 else torch.float32
        self.trocr = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-small-printed", torch_dtype=dtype).to(self.device)
        self.trocr.eval()
        self.batch_size = batch_size
        self.pad_x, self.pad_y = pad_x, pad_y
        print(f"  Device={self.device}  fp16={self.use_fp16}  batch={batch_size}")

    def _preprocess(self, crop):
        w, h = crop.size
        if w < 10 or h < 5:
            return None
        gray = crop.convert("L")
        gray = ImageOps.autocontrast(gray, cutoff=2)
        gray = gray.filter(ImageFilter.SHARPEN)
        gray = ImageOps.expand(
            gray, border=(self.pad_x, self.pad_y, self.pad_x, self.pad_y),
            fill=255)
        w2, h2 = gray.size
        if h2 < 64:
            scale = 64 / h2
            gray = gray.resize((max(int(w2 * scale), 1), 64), Image.LANCZOS)
        return gray.convert("RGB")

    def _ocr_batch(self, crops):
        texts = []
        for i in range(0, len(crops), self.batch_size):
            batch = crops[i: i + self.batch_size]
            pv = self.processor(images=batch, return_tensors="pt",
                                padding=True).pixel_values.to(self.device)
            if self.use_fp16:
                pv = pv.half()
            with torch.inference_mode():
                ids = self.trocr.generate(
                    pv, max_new_tokens=self.max_new_tokens,
                    num_beams=1, do_sample=False, use_cache=True)
            texts.extend(self.processor.batch_decode(ids, skip_special_tokens=True))
        return texts

    def detect(self, image):
        res = self.yolo(image, verbose=False, device=self.device, half=self.use_fp16)
        boxes = (res[0].boxes.xyxy.cpu().numpy()
                 if res[0].boxes is not None and len(res[0].boxes) else np.array([]))
        if len(boxes) < 3:
            w, h = image.size
            boxes = np.array([[0, y, w, min(y + 50, h)] for y in range(0, h, 50)],
                             dtype=float)
        return [tuple(map(int, b[:4])) for b in boxes]


# ── Cache builder with horizontal merging ───────────────────────────────────
class CacheBuilder:
    def __init__(self, runner, merger):
        self.runner = runner
        self.merger = merger

    def build(self, img_dir, track_per_image_timing=False):
        files = sorted(f for f in os.listdir(img_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        print(f"  {len(files)} images → YOLO + Merge + TrOCR")
        all_crops, image_meta = [], []
        t_yolo = t_pre = t_merge = 0.0
        per_image_pre_ocr = {}
        for i, fn in enumerate(files):
            img = Image.open(os.path.join(img_dir, fn)).convert("RGB")
            img_w, img_h = img.size
            base = fn.rsplit('.', 1)[0]

            ti0 = time.time()
            t0 = time.time()
            boxes = self.runner.detect(img)
            t_yolo += time.time() - t0

            t0 = time.time()
            boxes = self.merger.merge(boxes, img_w)
            t_merge += time.time() - t0

            my_idx, my_box = [], []
            t0 = time.time()
            for (x1, y1, x2, y2) in boxes:
                crop = self.runner._preprocess(img.crop((x1, y1, x2, y2)))
                if crop is None:
                    continue
                my_idx.append(len(all_crops))
                my_box.append((x1, y1, x2, y2))
                all_crops.append(crop)
            t_pre += time.time() - t0
            if track_per_image_timing:
                per_image_pre_ocr[base] = time.time() - ti0

            image_meta.append({"filename": fn, "base": base, "w": img_w, "h": img_h,
                               "crop_idx": my_idx, "boxes": my_box,
                               "n_crops": len(my_idx)})
            if (i + 1) % 50 == 0 or (i + 1) == len(files):
                print(f"    [detect {i+1}/{len(files)}]  crops={len(all_crops)}")

        print(f"  YOLO {t_yolo:.1f}s  |  Merge {t_merge:.1f}s  |  Preproc {t_pre:.1f}s")
        print(f"  OCR'ing {len(all_crops)} merged crops...")
        order = sorted(range(len(all_crops)), key=lambda j: all_crops[j].size[0])
        sorted_crops = [all_crops[j] for j in order]
        t0 = time.time()
        sorted_texts = self.runner._ocr_batch(sorted_crops)
        t_ocr = time.time() - t0
        print(f"  OCR total {t_ocr:.1f}s")

        texts = [None] * len(all_crops)
        for sp, op in enumerate(order):
            texts[op] = sorted_texts[sp]

        cache = {}
        for meta in image_meta:
            spatial = []
            for ci, (x1, y1, x2, y2) in zip(meta["crop_idx"], meta["boxes"]):
                t = texts[ci].strip()
                if not t:
                    continue
                cy, cx = (y1 + y2) / 2, (x1 + x2) / 2
                spatial.append({
                    "text": t, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "rel_y": cy / meta["h"], "rel_x": cx / meta["w"],
                    "w_ratio": (x2 - x1) / meta["w"],
                    "h_ratio": (y2 - y1) / meta["h"],
                    "x2_ratio": x2 / meta["w"],
                    "x1_ratio": x1 / meta["w"],
                })
            cache[meta["base"]] = {"filename": meta["filename"],
                                   "spatial_lines": spatial}

        per_image_total = {}
        if track_per_image_timing:
            total_crops = max(len(all_crops), 1)
            for meta in image_meta:
                share = meta["n_crops"] / total_crops
                ocr_share = t_ocr * share
                per_image_total[meta["base"]] = (
                    per_image_pre_ocr.get(meta["base"], 0.0) + ocr_share)
        return cache, per_image_total


# ── Feature extraction: TOTAL candidates ────────────────────────────────────
class TotalFeatureExtractor:
    PRICE = re.compile(
        r'(?:rm\s*|\$\s*)?(\d{1,6}(?:[.,]\d{3})*[.,]\d{2})(?!\d)(?!\s*%)',
        re.IGNORECASE)
    TIER1 = re.compile(
        r'\b(grand\s*total|total\s*amount|total\s*due|amount\s*due|'
        r'nett?\s*total|total\s*nett?|total\s*payment|total\s*rm|'
        r'jumlah\s*(besar|keseluruhan|akhir|bayar))\b', re.IGNORECASE)
    TIER2 = re.compile(r'(?<!\w)(total|jumlah|amaun|payable|to\s*pay)(?!\w)',
                       re.IGNORECASE)
    CASH  = re.compile(r'\b(cash|tunai|change|baki|deposit|tendered)\b',
                       re.IGNORECASE)

    @staticmethod
    def parse_float(s):
        s = re.sub(r'^rm\s*', '', s.strip(), flags=re.IGNORECASE).replace(',', '.')
        parts = s.split('.')
        if len(parts) > 2:
            s = ''.join(parts[:-1]) + '.' + parts[-1]
        try:
            return float(s)
        except Exception:
            return 0.0

    @staticmethod
    def fmt(s):
        return re.sub(r'^rm\s*', '', s.strip(), flags=re.IGNORECASE).replace(',', '.')

    def candidates(self, spatial_lines):
        if not spatial_lines:
            return []
        all_vals, per_line_prices = [], []
        line_has_tier1, line_has_tier2, line_has_cash = [], [], []
        for sl in spatial_lines:
            vals = []
            for m in self.PRICE.finditer(sl["text"]):
                v = self.parse_float(m.group(1))
                if v > 0:
                    vals.append(v); all_vals.append(v)
            per_line_prices.append(vals)
            line_has_tier1.append(bool(self.TIER1.search(sl["text"])))
            line_has_tier2.append(bool(self.TIER2.search(sl["text"])))
            line_has_cash.append(bool(self.CASH.search(sl["text"])))
        global_max = max(all_vals) if all_vals else 0.0
        n_tier1_total = sum(line_has_tier1)
        n_tier2_total = sum(line_has_tier2)
        # Sorted descending unique-ish values for rank
        sorted_vals = sorted(set(all_vals), reverse=True)
        val_rank_map = {v: i for i, v in enumerate(sorted_vals)}

        def _dist_above(idx, flags):
            for d in range(0, idx + 1):
                if flags[idx - d]:
                    return d
            return -1

        def _dist_below(idx, flags, n):
            for d in range(1, n - idx):
                if flags[idx + d]:
                    return d
            return -1

        rows = []
        n = len(spatial_lines)
        for idx, sl in enumerate(spatial_lines):
            text = sl["text"]
            line_vals = per_line_prices[idx]
            line_max = max(line_vals) if line_vals else 0.0
            d_tier1 = _dist_above(idx, line_has_tier1)
            d_tier2 = _dist_above(idx, line_has_tier2)
            d_cash  = _dist_above(idx, line_has_cash)
            d_tier1_b = _dist_below(idx, line_has_tier1, n)
            d_tier2_b = _dist_below(idx, line_has_tier2, n)
            for m in self.PRICE.finditer(text):
                price_str = m.group(1)
                val = self.parse_float(price_str)
                if val <= 0:
                    continue
                start = m.start()
                ctx_before = " ".join(s["text"] for s in spatial_lines[max(0, idx-3):idx])
                ctx_after  = " ".join(s["text"] for s in spatial_lines[idx+1:min(n, idx+4)])
                ctx = " ".join([ctx_before, text, ctx_after])
                rows.append({
                    "line_idx": idx,
                    "match_start": start,
                    "value": val,
                    "value_str": self.fmt(price_str),
                    "rel_y": float(sl["rel_y"]),
                    "x2_ratio": float(sl["x2_ratio"]),
                    "w_ratio": float(sl["w_ratio"]),
                    "h_ratio": float(sl["h_ratio"]),
                    "rel_x": float(sl["rel_x"]),
                    "has_tier1_keyword": int(bool(self.TIER1.search(text))),
                    "has_tier2_keyword": int(bool(self.TIER2.search(text))),
                    "has_tier1_in_ctx":  int(bool(self.TIER1.search(ctx))),
                    "has_cash_keyword":  int(bool(self.CASH.search(text))),
                    "has_cash_in_ctx":   int(bool(self.CASH.search(ctx))),
                    "price_string_pos_ratio": start / max(len(text), 1),
                    "is_largest_in_row": int(bool(line_vals) and val >= line_max - 1e-6),
                    "is_global_max":     int(global_max > 0 and val >= global_max - 1e-6),
                    "ratio_to_global_max": (val / global_max) if global_max > 0 else 0.0,
                    "n_prices_in_line":  len(line_vals),
                    "line_idx_norm":     idx / max(n - 1, 1),
                    "is_bottom_quarter": int(sl["rel_y"] > 0.75),
                    "log_val":           float(np.log1p(val)),
                    "dist_to_tier1_above": float(d_tier1) if d_tier1 >= 0 else 99.0,
                    "dist_to_tier2_above": float(d_tier2) if d_tier2 >= 0 else 99.0,
                    "dist_to_cash_above":  float(d_cash)  if d_cash  >= 0 else 99.0,
                    "is_below_tier1":      int(d_tier1 >= 0 and d_tier1 <= 2),
                    "is_below_tier2":      int(d_tier2 >= 0 and d_tier2 <= 2),
                    "is_below_cash":       int(d_cash  >= 0 and d_cash  <= 2),
                    "n_lines_total":       float(n),
                    # NEW iter2 features
                    "dist_to_tier1_below": float(d_tier1_b) if d_tier1_b >= 0 else 99.0,
                    "dist_to_tier2_below": float(d_tier2_b) if d_tier2_b >= 0 else 99.0,
                    "is_above_tier1":      int(d_tier1_b >= 0 and d_tier1_b <= 1),
                    "is_above_tier2":      int(d_tier2_b >= 0 and d_tier2_b <= 1),
                    "n_tier1_in_image":    float(n_tier1_total),
                    "n_tier2_in_image":    float(n_tier2_total),
                    "val_is_round":        int(abs(val * 100 - round(val * 100)) < 0.01
                                               and int(round(val * 100)) % 100 == 0),
                    "val_rank":            float(val_rank_map.get(val, 0)),
                    "val_rank_norm":       val_rank_map.get(val, 0) / max(len(sorted_vals) - 1, 1),
                    "is_top2_val":         int(val_rank_map.get(val, 999) < 2),
                    "frac_below_global_max": (1.0 - val / global_max) if global_max > 0 else 0.0,
                })
        return rows


def label_total(cands, gt_total):
    """Mark every candidate whose numeric value equals GT total as positive.
       (Multi-positive labeling — iter3 single-positive variant hurt CV.)"""
    gt_val = TotalFeatureExtractor.parse_float(gt_total) if gt_total else 0.0
    for c in cands:
        c["label"] = int(gt_val > 0 and abs(c["value"] - gt_val) < 1e-2)
    return cands


# ── Feature extraction: COMPANY candidates ──────────────────────────────────
class CompanyFeatureExtractor:
    COMP_KW = re.compile(
        r'\b(sdn\.?\s*bhd\.?|bhd\.?|sdn\.?|enterprise|marketing|trading|'
        r'industries?|holdings?|group|corporation|corp\.?|perniagaan|kedai|'
        r'restoran|restaurant|hotel|bakery|pharmacy|clinic|hospital|mart|'
        r'supermarket|minimarket|hypermarket|hardware|services?|supply|'
        r'supplier|petroleum|petrol|petron|shell|caltex|s/b|s\.b\.)\b',
        re.IGNORECASE)
    SKIP = re.compile(
        r'\b(copy|duplicate|tel|fax|email|www\.|\.com|\.my|gst|tax\s*invoice|'
        r'cash\s*bill|receipt|invoice|roc\s*no|reg\.?\s*no|formerly|licensee|'
        r'simplified|credit\s*note|debit\s*note)\b', re.IGNORECASE)
    PHONE = re.compile(r'\b(tel|fax|hp|phone)\b[:.\s]*\d', re.IGNORECASE)
    DIGITY = re.compile(r'\d')

    @staticmethod
    def _clean(s):
        s = re.sub(r'\s*\([^)]*\)\s*', ' ', s)
        return re.sub(r'\s+', ' ', s).strip(' ,.-')

    def candidates(self, spatial_lines):
        if not spatial_lines:
            return []
        rows = []
        n = len(spatial_lines)
        top_n = min(15, n)
        for i in range(top_n):
            sl = spatial_lines[i]
            text = sl["text"].strip()
            if len(text) < 2:
                continue
            cand_text = self._clean(text)
            if not cand_text:
                continue
            rows.append(self._feats(cand_text, sl, i, n, spatial_lines, is_merged=0))
            if i + 1 < top_n:
                nxt = spatial_lines[i + 1]
                ntxt = nxt["text"].strip()
                dy = nxt["rel_y"] - sl["rel_y"]
                if ntxt and 0 < dy < 0.06:
                    merged_text = self._clean(text + " " + ntxt)
                    if merged_text:
                        merged_sl = {
                            "rel_y": (sl["rel_y"] + nxt["rel_y"]) / 2,
                            "rel_x": (sl["rel_x"] + nxt["rel_x"]) / 2,
                            "w_ratio": max(sl["w_ratio"], nxt["w_ratio"]),
                            "h_ratio": sl["h_ratio"] + nxt["h_ratio"],
                            "x2_ratio": max(sl["x2_ratio"], nxt["x2_ratio"]),
                        }
                        rows.append(self._feats(merged_text, merged_sl, i, n,
                                                spatial_lines, is_merged=1))
        return rows

    def _feats(self, text, sl, idx, n, lines, is_merged):
        words = text.split()
        n_words = len(words)
        upper = sum(1 for w in words if w and (w.isupper() or w[0].isupper()))
        upper_ratio = upper / max(n_words, 1)
        digits = len(self.DIGITY.findall(text))
        has_comp_kw = int(bool(self.COMP_KW.search(text)))
        has_skip_kw = int(bool(self.SKIP.search(text)))
        has_phone   = int(bool(self.PHONE.search(text)))
        ctx_after = " ".join(s["text"] for s in lines[idx+1:min(n, idx+3)])
        has_comp_kw_below = int(bool(self.COMP_KW.search(ctx_after)))
        return {
            "value_str": text,
            "line_idx": idx,
            "is_merged": is_merged,
            "rel_y": float(sl["rel_y"]),
            "rel_x": float(sl["rel_x"]),
            "w_ratio": float(sl["w_ratio"]),
            "h_ratio": float(sl["h_ratio"]),
            "x2_ratio": float(sl["x2_ratio"]),
            "line_idx_norm": idx / max(n - 1, 1),
            "is_top_15":  int(sl["rel_y"] < 0.15),
            "is_top_25":  int(sl["rel_y"] < 0.25),
            "is_top_35":  int(sl["rel_y"] < 0.35),
            "n_words":    float(n_words),
            "char_len":   float(len(text)),
            "upper_ratio": float(upper_ratio),
            "n_digits":   float(digits),
            "digit_ratio": digits / max(len(text), 1),
            "has_comp_kw": has_comp_kw,
            "has_skip_kw": has_skip_kw,
            "has_phone":   has_phone,
            "has_comp_kw_below": has_comp_kw_below,
            "n_lines_total": float(n),
        }


def label_best_match(cands, gt_text, key_field="value_str", thr=0.7):
    """Mark the single candidate with highest token-F1 vs GT as label=1
       (if F1 >= thr); all others 0."""
    if not cands:
        return cands
    if not gt_text:
        for c in cands: c["label"] = 0
        return cands
    best_i, best_f1 = -1, 0.0
    for i, c in enumerate(cands):
        f1 = _token_f1(c.get(key_field, ""), gt_text)
        if f1 > best_f1:
            best_f1, best_i = f1, i
    for i, c in enumerate(cands):
        c["label"] = int(i == best_i and best_f1 >= thr)
    return cands


# ── Feature extraction: DATE candidates ─────────────────────────────────────
class DateFeatureExtractor:
    DATE_PATS = [
        (0, re.compile(r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})\b')),
        (1, re.compile(r'\b(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b')),
        (2, re.compile(r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2})\b')),
        (3, re.compile(r'\b(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|'
                       r'nov|dec)[a-z]*\s+\d{2,4})\b', re.IGNORECASE)),
    ]
    DATE_KW = re.compile(r'\b(date|tarikh|invoice\s*date|tax\s*invoice)\b',
                         re.IGNORECASE)

    def candidates(self, spatial_lines):
        if not spatial_lines:
            return []
        n = len(spatial_lines)
        line_has_kw = [bool(self.DATE_KW.search(s["text"])) for s in spatial_lines]
        # collect raw matches first to know total per image
        raw = []
        for idx, sl in enumerate(spatial_lines):
            for pid, pat in self.DATE_PATS:
                for m in pat.finditer(sl["text"]):
                    raw.append((idx, pid, m.start(), m.group(1)))
        n_dates = len(raw)
        rows = []
        for (idx, pid, start, val) in raw:
            sl = spatial_lines[idx]
            # distance to nearest date-keyword line (signed: above positive, below negative)
            d_above = -1
            for d in range(0, idx + 1):
                if line_has_kw[idx - d]:
                    d_above = d; break
            d_below = -1
            for d in range(1, n - idx):
                if line_has_kw[idx + d]:
                    d_below = d; break
            rows.append({
                "value_str": val,
                "line_idx": idx,
                "rel_y": float(sl["rel_y"]),
                "rel_x": float(sl["rel_x"]),
                "w_ratio": float(sl["w_ratio"]),
                "h_ratio": float(sl["h_ratio"]),
                "line_idx_norm": idx / max(n - 1, 1),
                "is_top_half":   int(sl["rel_y"] < 0.5),
                "is_top_25":     int(sl["rel_y"] < 0.25),
                "is_top_50":     int(sl["rel_y"] < 0.50),
                "pat_id":        float(pid),
                "pat_is_dmy4":   int(pid == 0),
                "pat_is_ymd":    int(pid == 1),
                "pat_is_dmy2":   int(pid == 2),
                "pat_is_text":   int(pid == 3),
                "n_dates_image": float(n_dates),
                "match_start":   float(start),
                "match_len":     float(len(val)),
                "match_pos_ratio": start / max(len(sl["text"]), 1),
                "has_date_kw_line": int(line_has_kw[idx]),
                "dist_to_kw_above": float(d_above) if d_above >= 0 else 99.0,
                "dist_to_kw_below": float(d_below) if d_below >= 0 else 99.0,
                "kw_within_2_above": int(0 <= d_above <= 2),
                "n_lines_total": float(n),
            })
        return rows


def label_date(cands, gt_date):
    gt_n = _norm(gt_date) if gt_date else ""
    if not gt_n:
        for c in cands: c["label"] = 0
        return cands
    for c in cands:
        c["label"] = int(_norm(c["value_str"]) == gt_n)
    # If none exact-matched, fallback to best token-F1 like address/company
    if not any(c["label"] for c in cands):
        return label_best_match(cands, gt_date, thr=0.5)
    return cands


# ── Feature extraction: ADDRESS candidates ──────────────────────────────────
class AddressFeatureExtractor:
    ADDR = re.compile(
        r'\b(jalan|jln|lot\s*\d|lorong|blk|block|road|taman|bandar|km\s*\d|'
        r'l/raya|lebuhraya|persiaran|lebuh|avenue|street|st\.?|kompleks)\b'
        r'|\b\d{5}\b', re.IGNORECASE)
    STOP = re.compile(
        r'\b(tel|fax|email|gst|total|date|tarikh|cashier|pelayan|thank|terima|'
        r'tax\s*invoice|cash\s*bill|invoice\s*no|receipt|reg\.?\s*no|roc\s*no|'
        r'www\.|\.com|\.my)\b', re.IGNORECASE)
    STATE = re.compile(
        r'\b(selangor|johor|penang|pulau\s*pinang|perak|kedah|kelantan|'
        r'pahang|terengganu|melaka|malacca|sabah|sarawak|negeri\s*sembilan|'
        r'perlis|kuala\s*lumpur|k\.?l\.?|wp|w\.p\.|putrajaya|labuan)\b',
        re.IGNORECASE)
    BLOCK = re.compile(
        r'\b(no\.?\s*\d|lot\s*\d|blok\s*\w|block\s*\w|tingkat|level|'
        r'unit\s*\w|suite\s*\w)\b', re.IGNORECASE)
    POSTCODE = re.compile(r'\b\d{5}\b')
    DIGITY = re.compile(r'\d')

    MAX_START = 30
    MAX_LEN   = 7

    def candidates(self, spatial_lines):
        if not spatial_lines:
            return []
        n = len(spatial_lines)
        line_addr = [bool(self.ADDR.search(s["text"])) for s in spatial_lines]
        line_stop = [bool(self.STOP.search(s["text"])) for s in spatial_lines]
        line_post = [bool(self.POSTCODE.search(s["text"])) for s in spatial_lines]
        line_state = [bool(self.STATE.search(s["text"])) for s in spatial_lines]
        line_block = [bool(self.BLOCK.search(s["text"])) for s in spatial_lines]
        rows = []
        top_start = min(self.MAX_START, n)
        for i in range(top_start):
            # Skip if starting line itself is a STOP-only header
            if line_stop[i]:
                continue
            for L in range(1, self.MAX_LEN + 1):
                j = i + L
                if j > n:
                    break
                if line_stop[j - 1]:
                    # extending past a stop is useless, but record the candidate before
                    break
                block = spatial_lines[i:j]
                texts = [b["text"].strip() for b in block]
                joined = ", ".join(t.rstrip(",").rstrip() for t in texts if t)
                joined = re.sub(r'\s+', ' ', joined).strip()
                if len(joined) < 3:
                    continue
                rel_ys  = [b["rel_y"] for b in block]
                w_rats  = [b["w_ratio"] for b in block]
                n_words = [len(b["text"].split()) for b in block]
                addr_hits = sum(line_addr[k] for k in range(i, j))
                post_hits = sum(line_post[k] for k in range(i, j))
                state_hits = sum(line_state[k] for k in range(i, j))
                block_hits = sum(line_block[k] for k in range(i, j))
                digit_lines = sum(bool(self.DIGITY.search(b["text"])) for b in block)
                comma_count = joined.count(",")
                rows.append({
                    "value_str":   joined,
                    "start_idx":   i,
                    "n_lines":     L,
                    "start_rel_y": float(rel_ys[0]),
                    "end_rel_y":   float(rel_ys[-1]),
                    "dy_span":     float(rel_ys[-1] - rel_ys[0]),
                    "avg_w_ratio": float(np.mean(w_rats)),
                    "max_w_ratio": float(np.max(w_rats)),
                    "avg_n_words": float(np.mean(n_words)),
                    "total_chars": float(len(joined)),
                    "avg_chars_per_line": len(joined) / L,
                    "start_idx_norm": i / max(n - 1, 1),
                    "is_top_15":    int(rel_ys[0] < 0.15),
                    "is_top_30":    int(rel_ys[0] < 0.30),
                    "is_top_50":    int(rel_ys[0] < 0.50),
                    "addr_hits":    float(addr_hits),
                    "frac_addr":    addr_hits / L,
                    "post_hits":    float(post_hits),
                    "has_postcode": int(post_hits > 0),
                    "frac_digit_lines": digit_lines / L,
                    "ends_at_postcode": int(line_post[j - 1]),
                    "n_lines_total":    float(n),
                    # NEW iter2 features
                    "state_hits":   float(state_hits),
                    "has_state":    int(state_hits > 0),
                    "block_hits":   float(block_hits),
                    "has_block":    int(block_hits > 0),
                    "first_line_is_block": int(line_block[i]),
                    "comma_count":  float(comma_count),
                    "comma_per_line": comma_count / L,
                    "ends_at_state": int(line_state[j - 1]),
                    "has_addr_and_post": int(addr_hits > 0 and post_hits > 0),
                    "has_addr_and_state": int(addr_hits > 0 and state_hits > 0),
                })
        return rows


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 64)
    print("  STAGE 1 — DATA PREP  |  YOLO + Merge + TrOCR + Features")
    print("=" * 64)

    train_pkl  = os.path.join(OUT_DIR, "spatial_cache_train.pkl")
    test_pkl   = os.path.join(OUT_DIR, "spatial_cache_test.pkl")
    timing_jp  = os.path.join(OUT_DIR, "test_timings.json")
    if os.path.exists(train_pkl) and os.path.exists(test_pkl):
        print("  >> Spatial caches already exist — skipping OCR rebuild")
        with open(train_pkl, "rb") as f: train_cache = pickle.load(f)
        with open(test_pkl,  "rb") as f: test_cache  = pickle.load(f)
        try:
            with open(timing_jp) as f: test_timings = json.load(f)
        except FileNotFoundError:
            test_timings = {}
    else:
        runner = FastYoloTrOCRRunner(YOLO_PATH, batch_size=128,
                                      use_fp16=True, max_new_tokens=32)
        merger = HorizontalBoxMerger(y_overlap_thr=0.5, x_gap_ratio=0.04)
        builder = CacheBuilder(runner, merger)

        print(f"\n  >> TRAIN split: {TRAIN_IMG_DIR}")
        t0 = time.time()
        train_cache, _ = builder.build(TRAIN_IMG_DIR, track_per_image_timing=False)
        print(f"  Train cache build wall: {time.time()-t0:.1f}s  |  images={len(train_cache)}")

        print(f"\n  >> TEST split: {TEST_IMG_DIR}")
        t0 = time.time()
        test_cache, test_timings = builder.build(TEST_IMG_DIR, track_per_image_timing=True)
        print(f"  Test cache build wall: {time.time()-t0:.1f}s  |  images={len(test_cache)}")

        del runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        with open(train_pkl, "wb") as f: pickle.dump(train_cache, f)
        with open(test_pkl,  "wb") as f: pickle.dump(test_cache,  f)
        with open(timing_jp, "w") as f:  json.dump(test_timings, f)

    train_bases = sorted(train_cache.keys())
    test_bases  = sorted(test_cache.keys())
    print(f"\n  Split: train={len(train_bases)}  test={len(test_bases)}")

    # ── Build candidate dataframes for ALL 4 keys ───────────────────────────
    tot_ex  = TotalFeatureExtractor()
    comp_ex = CompanyFeatureExtractor()
    date_ex = DateFeatureExtractor()
    addr_ex = AddressFeatureExtractor()

    for split_name, bases_, cache_, gt_dir in (
            ("train", train_bases, train_cache, TRAIN_GT_DIR),
            ("test",  test_bases,  test_cache,  TEST_GT_DIR)):

        rows_tot, rows_comp, rows_date, rows_addr = [], [], [], []
        for base in bases_:
            lines = cache_[base]["spatial_lines"]
            gt = load_gt(os.path.join(gt_dir, base + ".txt"))

            ct = label_total(tot_ex.candidates(lines), gt.get("total", ""))
            cc = label_best_match(comp_ex.candidates(lines), gt.get("company", ""), thr=0.7)
            cd = label_date(date_ex.candidates(lines), gt.get("date", ""))
            ca = label_best_match(addr_ex.candidates(lines), gt.get("address", ""), thr=0.55)

            for r in ct:  r["base"] = base; rows_tot.append(r)
            for r in cc:  r["base"] = base; rows_comp.append(r)
            for r in cd:  r["base"] = base; rows_date.append(r)
            for r in ca:  r["base"] = base; rows_addr.append(r)

        def _save(rows, fname):
            pos = sum(int(r["label"]) for r in rows)
            n_img_with_pos = len({r["base"] for r in rows if r["label"]})
            print(f"  {split_name}/{fname}: cands={len(rows)}  pos={pos}  "
                  f"imgs_with_pos={n_img_with_pos}")
            pd.DataFrame(rows).to_parquet(os.path.join(OUT_DIR, fname), index=False)

        _save(rows_tot,  f"candidates_total_{split_name}.parquet")
        _save(rows_comp, f"candidates_company_{split_name}.parquet")
        _save(rows_date, f"candidates_date_{split_name}.parquet")
        _save(rows_addr, f"candidates_address_{split_name}.parquet")

    with open(os.path.join(OUT_DIR, "split_meta.json"), "w") as f:
        json.dump({"train_bases": train_bases, "test_bases": test_bases}, f)

    print("  Saved: spatial caches, 4×{train,test} candidate parquets, "
          "test_timings.json, split_meta.json")
    print("=" * 64)


if __name__ == "__main__":
    main()
