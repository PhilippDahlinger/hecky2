import cv2
import os
from pathlib import Path

# Paths
images_dir = Path("yolo_dataset/images")
labels_dir = Path("yolo_dataset/labels")
output_dir = Path("visualized")
output_dir.mkdir(exist_ok=True)

# Optional: load class names
class_names = ["dice"]  # Update with real class names if you have them


def visualize_yolo(image_path):
    label_path = labels_dir / (image_path.stem + ".txt")
    image = cv2.imread(str(image_path))
    h, w, _ = image.shape

    # Draw boxes
    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, cx, cy, bw, bh = map(float, parts)
                cls_id = int(cls_id)

                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                color = (0, 255, 0)
                label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Save or show
    out_path = output_dir / image_path.name
    cv2.imwrite(str(out_path), image)
    print(f"Saved: {out_path}")

# Example: visualize all images
for img_file in images_dir.glob("*.jpg"):
    visualize_yolo(img_file)
