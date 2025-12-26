import os
import json
from pathlib import Path
import glob
import shutil
import cv2

from configs import RAW_DATA_DIR, CROPS_DIR, CLASSES_FILE
from utils import crop_and_save, ensure_dir, list_image_files

def _load_names_map():
    if CLASSES_FILE and os.path.exists(CLASSES_FILE):
        with open(CLASSES_FILE, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        return {i: name for i, name in enumerate(lines)}
    return None

def _parse_yolo_txt(txt_path, img_w, img_h, names_map=None):
    boxes = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h
            label = str(cls) if names_map is None else names_map.get(cls, str(cls))
            boxes.append((label, [x1, y1, x2, y2]))
    return boxes

def _parse_coco_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    images = {img['id']: img for img in coco.get('images', [])}
    cats = {c['id']: c['name'] for c in coco.get('categories', [])}
    anns = coco.get('annotations', [])
    rows = []
    base_dir = os.path.dirname(json_path)
    for ann in anns:
        img = images[ann['image_id']]
        x, y, w, h = ann['bbox']
        x1, y1, x2, y2 = x, y, x + w, y + h
        rows.append({
            'image_path': os.path.join(base_dir, img['file_name']),
            'bbox': [x1, y1, x2, y2],
            'label': cats[ann['category_id']]
        })
    return rows

def main():
    ensure_dir(CROPS_DIR)
    names_map = _load_names_map()
    coco_files = list(Path(RAW_DATA_DIR).glob('*.json'))
    if coco_files:
        rows = _parse_coco_json(str(coco_files[0]))
        for r in rows:
            img_path = r['image_path']
            base = Path(img_path).stem
            outname = f"{base}_crop.jpg"
            outdir = os.path.join(CROPS_DIR, r['label'])
            ensure_dir(outdir)
            outpath = os.path.join(outdir, outname)
            try:
                crop_and_save(img_path, r['bbox'], outpath)
            except Exception:
                continue
        print('Crops created in', CROPS_DIR)
        return

    image_files = list_image_files(RAW_DATA_DIR)
    if not image_files:
        print('No images found in', RAW_DATA_DIR)
        return

    for img_path in image_files:
        txt_path = os.path.splitext(img_path)[0] + '.txt'
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        if not os.path.exists(txt_path):
            outdir = os.path.join(CROPS_DIR, 'unknown')
            ensure_dir(outdir)
            outpath = os.path.join(outdir, Path(img_path).name)
            shutil.copy(img_path, outpath)
            continue
        boxes = _parse_yolo_txt(txt_path, w, h, names_map=names_map)
        base = Path(img_path).stem
        for i, (label, bbox) in enumerate(boxes):
            outdir = os.path.join(CROPS_DIR, label)
            ensure_dir(outdir)
            outname = f"{base}_crop_{i}.jpg"
            outpath = os.path.join(outdir, outname)
            try:
                crop_and_save(img_path, bbox, outpath)
            except Exception:
                continue
    print('Crops created in', CROPS_DIR)

if __name__ == '__main__':
    main()