import os
import cv2
from PIL import Image
from pathlib import Path
from tflite_detector import Detector

# Config
video_dir = Path("/home/philipp/projects/flutter_projects/rescue_hecky/negative_samples_videos")  # Folder with multiple .mp4 or .mov files
negative_samples = True
output_dir = Path("yolo_dataset_negative_samples")
images_dir = output_dir / "images"
labels_dir = output_dir / "labels"
model_path = "/home/philipp/projects/flutter_projects/rescue_hecky/extracted_apk/assets/model.tflite"
label_path = "/home/philipp/projects/flutter_projects/rescue_hecky/extracted_apk/assets/labels.txt"
frame_skip = 5  # Use every 5th frame

# Setup output directories
images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

# Initialize model
detector = Detector(model_path=model_path, label_path=label_path)

# Process all video files
for video_file in video_dir.glob("*.mp4"):
    print(f"Processing video: {video_file.name}")
    cap = cv2.VideoCapture(str(video_file))
    base_name = video_file.stem
    frame_index = 0
    saved_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        print(frame_index)
        if not ret:
            break

        if frame_index % frame_skip != 0:
            frame_index += 1
            continue

        # Convert and detect
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes = detector.detect(pil_image)

        if negative_samples:
            label_name = f"{base_name}_frame_{saved_index:05d}.txt"
            img_name = f"{base_name}_frame_{saved_index:05d}.jpg"
            with open(labels_dir / label_name, "w") as f:
                # write empty label file for negative samples
                f.write("")
            pil_image.save(images_dir / img_name)
            saved_index += 1

        else:
            if boxes and len(boxes) <= 8:
                print("Detected boxes:", len(boxes))
                img_name = f"{base_name}_frame_{saved_index:05d}.jpg"
                label_name = f"{base_name}_frame_{saved_index:05d}.txt"

                pil_image.save(images_dir / img_name)

                img_h, img_w = pil_image.height, pil_image.width
                with open(labels_dir / label_name, "w") as f:
                    for box in boxes:
                        f.write(f"{box.cls} {box.cx:.6f} {box.cy:.6f} {box.w:.6f} {box.h:.6f}\n")

                saved_index += 1

        frame_index += 1

    cap.release()
    print(f"✔ Processed {video_file.name} — saved {saved_index} frames.")

print("✅ All videos processed.")
