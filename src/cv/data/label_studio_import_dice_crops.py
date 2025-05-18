import os, json, glob

CROP_DIR = "data/dice_crops"
OUT_JSON = "label_studio_import_dice_crops.json"

tasks = []
for ext in ("*.jpg", "*.jpeg", "*.png"):
    for path in glob.glob(os.path.join(CROP_DIR, ext)):
        fname = os.path.basename(path)
        fname = "http://localhost:8888/" + fname  # use absolute filename
        tasks.append({"data": {"image": fname}})

with open(OUT_JSON, "w") as f:
    json.dump(tasks, f, indent=2)

print(f"Created {OUT_JSON} with {len(tasks)} tasks.")
