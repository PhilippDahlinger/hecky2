import os
import json
import uuid

BASE_URL = "http://localhost:8888"  # your HTTP server URL
BASE_DIR = "data/labeled_frames"  # your folders with pred.json + raw_image.jpg


def generate_id():
    return str(uuid.uuid4())


def normalize_label(label):
    # returns label name (e.g. "dice") and confidence as float if possible, else 1.0
    if "(" in label and ")" in label:
        try:
            base = label.split("(")[0]
            conf = float(label.split("(")[1].split(")")[0])
            return base, conf
        except Exception:
            pass
    return label, 1.0


output = []

for folder_name in os.listdir(BASE_DIR):
    folder_path = os.path.join(BASE_DIR, folder_name)
    pred_path = os.path.join(folder_path, "pred.json")
    image_url = f"{BASE_URL}/{folder_name}/raw_image.jpg"

    if not os.path.exists(pred_path):
        continue

    with open(pred_path, "r") as f:
        data = json.load(f)

    boxes = data["boxes"]
    labels = data["labels"]
    w, h = data["size"]

    results = []
    for i, (box, label) in enumerate(zip(boxes, labels)):
        clean_label, confidence = normalize_label(label)

        # Filter by confidence threshold
        if confidence < 0.3:
            continue

        cx, cy, bw, bh = box
        abs_w = bw * w
        abs_h = bh * h

        # Filter by max dimension <= 1/4 image size
        if max(abs_w / w, abs_h / h) > 0.25:
            continue

        abs_x = (cx - bw / 2) * w
        abs_y = (cy - bh / 2) * h

        # Convert to percentages for Label Studio
        x_perc = abs_x * 100 / w
        y_perc = abs_y * 100 / h
        w_perc = abs_w * 100 / w
        h_perc = abs_h * 100 / h

        results.append({
            "id": generate_id(),
            "type": "rectanglelabels",
            "from_name": "label",
            "to_name": "image",
            "original_width": w,
            "original_height": h,
            "image_rotation": 0,
            "value": {
                "rotation": 0,
                "x": x_perc,
                "y": y_perc,
                "width": w_perc,
                "height": h_perc,
                "rectanglelabels": [clean_label]
            }
        })

    output.append({
        "data": {
            "image": image_url
        },
        "predictions": [{
            "model_version": "groundingdino",
            "result": results
        }]
    })

# Write output to JSON file for Label Studio import
with open("label_studio_import.json", "w") as out_file:
    json.dump(output, out_file, indent=2)

print("Export complete: label_studio_import.json")
