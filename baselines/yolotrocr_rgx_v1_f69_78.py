"""
YT-Rex Baseline Pipeline — YOLO + TrOCR + Simple Keyword Regex v1
Avg F1: 69.78%  |  N=347 SROIE test set
"""

import os, json, time, re
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter, ImageOps
from collections import Counter
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

YOLO_PATH = "/kaggle/input/models/muhammedeminadyaman/yolov8/pytorch/default/1/best.pt"
IMG_DIR   = "/kaggle/input/datasets/urbikn/sroie-datasetv2/SROIE2019/test/img"
GT_DIR    = "/kaggle/input/datasets/urbikn/sroie-datasetv2/SROIE2019/test/entities"
KEYS      = ["company", "date", "address", "total"]


def _clean_price(raw: str) -> str:
    raw = raw.strip()
    if re.fullmatch(r'\d+[.,]\d{2}', raw):
        return raw.replace(',', '.')
    m = re.fullmatch(r'(\d{1,3}(?:[.,]\d{3})+)[.,](\d{2})', raw)
    if m:
        return re.sub(r'[.,]', '', m.group(1)) + '.' + m.group(2)
    return raw.replace(',', '.')


class YoloOcrPipeline:
    def __init__(self, yolo_path="best.pt"):
        print("Loading models...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {self.device.upper()}")
        self.yolo = YOLO(yolo_path)
        model_name = "microsoft/trocr-small-printed"
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.trocr = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.trocr.eval()
        self.batch_size = 8
        print("  Models ready.\n")

    def _preprocess_crop(self, crop):
        w, h = crop.size
        if w < 10 or h < 5:
            return None
        if h < 64:
            scale = 64 / h
            crop = crop.resize((max(int(w * scale), 1), 64), Image.LANCZOS)
        gray = crop.convert("L")
        gray = ImageOps.autocontrast(gray, cutoff=2)
        gray = gray.filter(ImageFilter.SHARPEN)
        return gray.convert("RGB")

    def _strip_boxes(self, image, strip_h=50):
        w, h = image.size
        boxes = []
        y = 0
        while y < h:
            boxes.append([0, y, w, min(y + strip_h, h)])
            y += strip_h
        return np.array(boxes, dtype=float)

    def _ocr_crops(self, crops):
        texts = []
        for i in range(0, len(crops), self.batch_size):
            batch = crops[i: i + self.batch_size]
            pv = self.processor(images=batch, return_tensors="pt",
                                padding=True).pixel_values.to(self.device)
            with torch.no_grad():
                ids = self.trocr.generate(pv, max_new_tokens=64, num_beams=1)
            texts.extend(self.processor.batch_decode(ids, skip_special_tokens=True))
        return texts

    def _extract_fields(self, full_text, lines):
        # ── DATE ──
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

        # ── TOTAL ──
        total = ""
        m = re.search(
            r'(?:grand\s*total|total\s*amount|total\s*due|total|amount\s*due)'
            r'[^\d]*(\d[\d,\.]+)',
            full_text, re.IGNORECASE)
        if m:
            total = _clean_price(m.group(1))
        else:
            prices = re.findall(r'\b\d{1,6}[.,]\d{2}\b', full_text)
            if prices:
                pf = []
                for p in prices:
                    try:
                        pf.append(float(p.replace(',', '.')))
                    except Exception:
                        pass
                if pf:
                    total = prices[pf.index(max(pf))].replace(',', '.')

        # ── COMPANY ──
        company = ""
        for line in lines[:8]:
            ls = line.strip()
            if not ls:
                continue
            if re.fullmatch(r'[\d\s\.\-\:\/\(\)]+', ls):
                continue
            if re.search(r'\b(jalan|jln|lot|no\.|blk|floor|level|tel|phone|fax|www|http)\b',
                         ls, re.IGNORECASE):
                continue
            company = ls
            break

        # ── ADDRESS ──
        ADDR_KW = re.compile(
            r'\b(jalan|jln|lot|no\.|block|blk|floor|level|road|street|avenue|'
            r'highway|taman|bandar|off|km\b|kompleks|plaza|tower|building|\d{5})\b',
            re.IGNORECASE)
        addr_lines = [line.strip() for line in lines if ADDR_KW.search(line)]
        address = " ".join(addr_lines[:4])

        return {"company": company, "date": date, "address": address, "total": total}

    def process_image(self, image):
        start = time.time()
        results = self.yolo(image, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) < 3:
            boxes = self._strip_boxes(image)
        else:
            boxes = boxes[boxes[:, 1].argsort()]

        crops, valid = [], []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.size[0], x2), min(image.size[1], y2)
            crop = self._preprocess_crop(image.crop((x1, y1, x2, y2)))
            if crop is not None:
                crops.append(crop)
                valid.append((x1, y1, x2, y2))

        texts = self._ocr_crops(crops) if crops else []
        lines = [t.strip() for t in texts if t.strip()]
        full_text = "\n".join(lines)
        extracted = self._extract_fields(full_text, lines)
        return extracted, time.time() - start


# ── Metrics ──────────────────────────────────────────────────────────────────

def normalize_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip().replace(',', '.')
    text = re.sub(r'[^\w\s\.]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def calculate_f1(pred, truth):
    p, t = normalize_text(pred).split(), normalize_text(truth).split()
    common = Counter(p) & Counter(t)
    same = sum(common.values())
    if not p or not t:
        return float(p == t)
    if same == 0:
        return 0.0
    return (2 * same / len(p) * same / len(t)) / (same / len(p) + same / len(t))

def calculate_em(pred, truth):
    return int(normalize_text(pred) == normalize_text(truth))

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


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipe = YoloOcrPipeline(yolo_path=YOLO_PATH)

    image_files = sorted(f for f in os.listdir(IMG_DIR)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    print(f"{len(image_files)} test images...\n")

    rows = []
    for i, fn in enumerate(image_files):
        img = Image.open(os.path.join(IMG_DIR, fn)).convert("RGB")
        base = fn.rsplit('.', 1)[0]
        gt = load_gt(os.path.join(GT_DIR, base + ".txt"))
        pred, lat = pipe.process_image(img)
        row = {"latency_sec": lat}
        for k in KEYS:
            row[f"{k}_f1"] = calculate_f1(pred.get(k, ""), gt.get(k, ""))
            row[f"{k}_em"] = calculate_em(pred.get(k, ""), gt.get(k, ""))
        rows.append(row)
        if (i + 1) % 50 == 0 or (i + 1) == len(image_files):
            print(f"  [{i+1}/{len(image_files)}]")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 55)
    print("  YOLO + TrOCR + Simple Regex v1  RESULTS")
    print("=" * 55)
    f1s, ems = [], []
    for k in KEYS:
        f1 = df[f"{k}_f1"].mean() * 100
        em = df[f"{k}_em"].mean() * 100
        f1s.append(f1); ems.append(em)
        print(f"  {k.upper():<10} F1: {f1:6.2f}%   EM: {em:6.2f}%")
    print(f"  {'AVERAGE':<10} F1: {sum(f1s)/len(f1s):6.2f}%   EM: {sum(ems)/len(ems):6.2f}%")
    avg_lat = df["latency_sec"].mean()
    print(f"  Speed: {avg_lat:.3f} s/img  ({1/avg_lat:.2f} FPS)")
    print("=" * 55)

    df.to_csv("/kaggle/working/v1_baseline_results.csv", index=False)
    print("\nSaved: /kaggle/working/v1_baseline_results.csv")
