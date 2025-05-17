from ultralytics import YOLO
import cv2

# Load weights
model = YOLO("runs/detect/train/weights/best.pt")

# img_path = "/home/philipp/projects/hecky2/data/exported_labels/yolo_with_images/dataset/images/val/0b91378c__raw_image.jpg"
img_path = "/home/philipp/projects/hecky2/data/demo.JPEG"

# Inference
results = model(img_path)  # list[ultralytics.engine.results.Results]

# Draw detections on the image
annotated = results[0].plot()  # numpy array (BGR)

# Show or save
# cv2.imshow("Prediction", annotated)
# cv2.waitKey(0)
cv2.imwrite("output_with_boxes.jpg", annotated)
