#!/usr/bin/env python3
# crop_detections.py
import os, glob
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# -------------------- SETTINGS --------------------
WEIGHTS = "runs/detect/train/weights/best.pt"  # your trained model
SOURCE = "data/accepted_frames"  # folder with images (recurses)
OUTDIR = "data/dice_crops"  # where to save crops
ENLARGE = 0.10  # 10 % padding around box
IMG_EXT = ("*.jpg", "*.jpeg", "*.png")


# --------------------------------------------------

def enlarge_box(box, w, h, ratio=0.10):
    """xyxy tensor/list → enlarged and clipped ints"""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    padw, padh = bw * ratio, bh * ratio
    nx1 = max(int(x1 - padw), 0)
    ny1 = max(int(y1 - padh), 0)
    nx2 = min(int(x2 + padw), w - 1)
    ny2 = min(int(y2 + padh), h - 1)
    return nx1, ny1, nx2, ny2


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    model = YOLO(WEIGHTS)
    img_paths = []
    for ext in IMG_EXT:
        img_paths += glob.glob(os.path.join(SOURCE, "**", ext), recursive=True)
    img_paths.sort()

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[skip] cannot read {img_path}")
            continue

        h, w = img.shape[:2]
        name = Path(img_path).stem

        # run detector
        results = model(img_path, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()  # (n,4)
        confs = results.boxes.conf.cpu().numpy()  # (n,)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = enlarge_box(box, w, h, ENLARGE)
            crop = img[y1:y2, x1:x2]
            out_name = f"{name}_{i}.jpg"
            cv2.imwrite(os.path.join(OUTDIR, out_name), crop)

        print(f"{name}: saved {len(boxes)} crops")


if __name__ == "__main__":
    main()
