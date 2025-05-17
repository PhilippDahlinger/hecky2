from ultralytics import YOLO

# 1️⃣  Path to your data.yaml (edit this)
DATA_YAML = "/home/philipp/projects/hecky2/data/exported_labels/yolo_with_images/dataset/data.yaml"   # contains train/val paths + names

# 2️⃣  Load the YOLO model (nano variant)
model = YOLO("model_checkpoints/yolo11n.pt")

# 3️⃣  Train on CPU
results = model.train(
    data=DATA_YAML,   # your dice dataset
    epochs=5,
    imgsz=640,
    device="cpu",     # force CPU
    batch=8,          # keep batch small for CPU
    workers=0         # avoid dataloader overhead on single-CPU
)

# Optional: save metrics
print(results)
