#!/usr/bin/env python3
# convert_and_split.py
import os
import glob
import random
import shutil

# ------------------------------------------------------------------
# SETTINGS
root = "/home/philipp/projects/hecky2/data/exported_labels/yolo_with_images"  # project root containing images/ and labels/
img_dir_in = os.path.join(root, "images")
lbl_dir_in = os.path.join(root, "labels")  # Label-Studio txts (x1 y1 x3 y3 …)
out_root = os.path.join(root, "dataset")  # output dataset
train_pct = 0.8  # 80 % train / 20 % val
seed = 42  # reproducibility


# ------------------------------------------------------------------

def ls_to_yolo(line):
    """Convert one Label-Studio rectangle line → YOLO string."""
    parts = list(map(float, line.strip().split()))
    cls, x1, y1, x2, y2, x3, y3, x4, y4 = parts
    xc = (x1 + x3) / 2
    yc = (y1 + y3) / 2
    w = abs(x3 - x1)
    h = abs(y3 - y1)
    return f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"


def convert_labels(src_txt, dst_txt):
    with open(src_txt) as f:
        lines_in = f.readlines()
    lines_out = [ls_to_yolo(l) for l in lines_in if l.strip()]
    with open(dst_txt, "w") as f:
        f.writelines(lines_out)


def main():
    random.seed(seed)

    # Gather all image files (jpg / png / jpeg)
    exts = ("*.jpg", "*.jpeg", "*.png")
    imgs = []
    for e in exts:
        imgs.extend(glob.glob(os.path.join(img_dir_in, e)))
    imgs.sort()

    # Create output dirs
    for split in ("train", "val"):
        os.makedirs(os.path.join(out_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_root, "labels", split), exist_ok=True)

    # Shuffle & split
    random.shuffle(imgs)
    cut = int(len(imgs) * train_pct)
    split_map = {p: "train" if i < cut else "val" for i, p in enumerate(imgs)}

    # Process each file
    for img_path, split in split_map.items():
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_in = os.path.join(lbl_dir_in, base + ".txt")
        if not os.path.isfile(lbl_in):
            print(f"[warn] missing label for {img_path} — skipped")
            continue

        # Copy image
        shutil.copy(img_path, os.path.join(out_root, "images", split, os.path.basename(img_path)))

        # Convert & save label
        lbl_out = os.path.join(out_root, "labels", split, base + ".txt")
        convert_labels(lbl_in, lbl_out)

    print(
        f"Done!  Train: {sum(v == 'train' for v in split_map.values())}  |  Val: {sum(v == 'val' for v in split_map.values())}")


if __name__ == "__main__":
    main()
