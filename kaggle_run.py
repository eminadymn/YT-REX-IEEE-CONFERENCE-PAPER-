"""
Kaggle helper for the YT-Rex pipeline.

Pushes 01_data_preparation.py + 02_model_training.py + 03_error_taxonomy.py
as a single notebook to the configured kernel, polls until completion, then
pulls the run's stdout into kernel_log.txt.

Usage (from this directory):
    export KAGGLE_API_TOKEN=KGAT_...   # Kaggle > Settings > API
    python kaggle_run.py push     # push only
    python kaggle_run.py wait     # poll until COMPLETE/ERROR/CANCEL
    python kaggle_run.py log      # download stdout into kernel_log.txt
    python kaggle_run.py all      # push + wait + log

Requirements: `pip install kaggle`, and the KAGGLE_API_TOKEN env var
(KGAT_-style Bearer token). Set USER / SLUG below to your own account and
kernel, and point MODELS at wherever the fine-tuned YOLOv8 weights live.
"""

import sys, os, json, re, time

if not os.environ.get("KAGGLE_API_TOKEN"):
    sys.exit("KAGGLE_API_TOKEN is not set (Kaggle > Settings > API).")

from kaggle import api  # noqa: E402
from kagglesdk.kernels.types.kernels_api_service import (  # noqa: E402
    ApiSaveKernelRequest,
    ApiGetKernelSessionStatusRequest,
    ApiListKernelSessionOutputRequest,
)

ROOT        = os.path.dirname(os.path.abspath(__file__))
USER        = "muhammedeminadyaman"
SLUG        = "yolo-ocr-regexws-donut"
TITLE       = "yolo-ocr-regexws-donut"
DATASETS    = ["urbikn/sroie-datasetv2"]
MODELS      = ["muhammedeminadyaman/yolov8/pytorch/default/1"]


def _strip_main(src: str, new_name: str) -> str:
    src = re.sub(r"\bdef\s+main\s*\(", f"def {new_name}(", src)
    src = re.sub(
        r"if\s+__name__\s*==\s*['\"]__main__['\"]\s*:\s*\n\s*main\(\)\s*",
        "", src)
    return src


def build_notebook_source() -> str:
    """Combine stage 1 + stage 2 into a single notebook cell.

    Includes a stale-cache cleanup preamble so each run regenerates the
    spatial cache from scratch (avoids picking up old fp32 artifacts).
    """
    with open(os.path.join(ROOT, "01_data_preparation.py")) as f:
        s1 = _strip_main(f.read(), "_stage1_main")
    with open(os.path.join(ROOT, "02_model_training.py")) as f:
        s2 = _strip_main(f.read(), "_stage2_main")
    with open(os.path.join(ROOT, "03_error_taxonomy.py")) as f:
        s3 = _strip_main(f.read(), "_stage3_main")
    # Keep spatial_cache_*.pkl across runs (OCR is expensive; downstream
    # candidate features re-derive from cache). Wipe everything else.
    preamble = (
        "import os, glob\n"
        "_KEEP = {'spatial_cache_train.pkl', 'spatial_cache_test.pkl', "
        "'test_timings.json', 'split_meta.json'}\n"
        "for _p in (glob.glob('/kaggle/working/*.pkl') + "
        "glob.glob('/kaggle/working/*.parquet') + "
        "glob.glob('/kaggle/working/*.json') + "
        "glob.glob('/kaggle/working/*.csv')):\n"
        "    if os.path.basename(_p) in _KEEP: continue\n"
        "    try: os.remove(_p); print(' removed stale:', _p)\n"
        "    except Exception: pass\n"
    )
    return (
        preamble
        + "\n# === STAGE 1 ===\n" + s1
        + "\n# === STAGE 2 ===\n" + s2
        + "\n# === STAGE 3 ===\n" + s3
        + "\nprint('\\n>>> RUNNING STAGE 1 <<<\\n')\n_stage1_main()\n"
        + "print('\\n>>> RUNNING STAGE 2 <<<\\n')\n_stage2_main()\n"
        + "print('\\n>>> RUNNING STAGE 3 <<<\\n')\n_stage3_main()\n"
        + "print('\\n>>> ALL DONE <<<')\n"
    )


def push() -> None:
    code = build_notebook_source()
    nb = {
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3",
                           "language": "python"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4, "nbformat_minor": 4,
        "cells": [{"cell_type": "code", "execution_count": None,
                   "metadata": {}, "outputs": [], "source": code}],
    }
    api.authenticate()
    with api.build_kaggle_client() as kc:
        req = ApiSaveKernelRequest()
        req.slug                     = f"{USER}/{SLUG}"
        req.new_title                = TITLE
        req.text                     = json.dumps(nb)
        req.language                 = "python"
        req.kernel_type              = "notebook"
        req.is_private               = True
        req.enable_gpu               = True
        req.enable_internet          = True
        req.dataset_data_sources     = DATASETS
        req.kernel_data_sources      = []
        req.competition_data_sources = []
        req.model_data_sources       = MODELS
        result = kc.kernels.kernels_api_client.save_kernel(req)
        print("version:", result.version_number)
        print("error:  ", result.error or "(none)")


def status() -> str:
    api.authenticate()
    with api.build_kaggle_client() as kc:
        sreq = ApiGetKernelSessionStatusRequest()
        sreq.user_name   = USER
        sreq.kernel_slug = SLUG
        s = kc.kernels.kernels_api_client.get_kernel_session_status(sreq)
        return str(s.status)


def wait(poll_seconds: int = 30) -> None:
    while True:
        s = status()
        print("STATUS:", s)
        if any(k in s for k in ("COMPLETE", "ERROR", "CANCEL")):
            return
        time.sleep(poll_seconds)


def fetch_log() -> None:
    api.authenticate()
    with api.build_kaggle_client() as kc:
        oreq = ApiListKernelSessionOutputRequest()
        oreq.user_name   = USER
        oreq.kernel_slug = SLUG
        res = kc.kernels.kernels_api_client.list_kernel_session_output(oreq)
        text = res.log if isinstance(res.log, str) else "".join(res.log)
        entries = json.loads(text)
        stdout = "".join(e.get("data", "") for e in entries
                         if e.get("stream_name") == "stdout")
        with open(os.path.join(ROOT, "kernel_log.txt"), "w") as f:
            f.write(stdout)
        print(f"wrote kernel_log.txt ({len(stdout)} chars)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if   cmd == "push": push()
    elif cmd == "wait": wait()
    elif cmd == "log":  fetch_log()
    elif cmd == "all":
        push(); wait(); fetch_log()
    else:
        print(__doc__); sys.exit(2)
