"""Shared constants and helpers for the SANPO (dataset 2) preparation scripts."""

import json
import os
import urllib.request
from pathlib import Path

BUCKET = "https://storage.googleapis.com/gresearch/sanpo_dataset/v0/sanpo-real"
LIST_API = "https://storage.googleapis.com/storage/v1/b/gresearch/o"

# Raw downloads live outside the repository (large); processed data goes to DDPM/data_sanpo/
RAW_DIR = Path(os.environ.get("SANPO_RAW_DIR", "/home/woody/iwi5/iwi5388h/datasets/sanpo_raw"))
DDPM_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(os.environ.get("SANPO_OUT_DIR", DDPM_ROOT / "data_sanpo"))
HERE = Path(__file__).parent

CAMERA = "camera_chest/left"  # chest-mounted, left stereo image: the pedestrian viewpoint


def fetch(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # proxy comes from https_proxy
        return r.read()


def fetch_json(url: str):
    return json.loads(fetch(url))
