import os
from pathlib import Path
import cv2
import numpy as np

from configs import IMAGE_SIZE

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def crop_and_save(img_path: str, bbox: list, out_path: str):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop_resized = cv2.resize(crop, IMAGE_SIZE)

    ensure_dir(os.path.dirname(out_path) or ".")
    saved = cv2.imwrite(out_path, crop_resized)
    return out_path if saved else None

def list_image_files(folder: str, exts=None):
    if exts is None:
        exts = ['.jpg', '.jpeg', '.png']
    files = []
    for root, _, filenames in os.walk(folder):
        for fn in filenames:
            if any(fn.lower().endswith(e) for e in exts):
                files.append(os.path.join(root, fn))
    files.sort()
    return files

def read_image_rgb(path: str):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image_rgb(path: str, img_rgb: np.ndarray):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ensure_dir(os.path.dirname(path) or ".")
    return cv2.imwrite(path, img_bgr)