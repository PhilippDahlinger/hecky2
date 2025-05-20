from ultralytics import YOLO
import cv2

img_path = "/home/philipp/projects/hecky2/data/demo.JPEG"


def pytorch_demo():
    # Load weights
    model = YOLO("runs/detect/train/weights/best.pt")

    # img_path = "/home/philipp/projects/hecky2/data/exported_labels/yolo_with_images/dataset/images/val/0b91378c__raw_image.jpg"

    # Inference
    results = model(img_path)  # list[ultralytics.engine.results.Results]

    # Draw detections on the image
    annotated = results[0].plot()  # numpy array (BGR)

    # Show or save
    # cv2.imshow("Prediction", annotated)
    # cv2.waitKey(0)
    cv2.imwrite("output_with_boxes.jpg", annotated)


def torchscript_demo():
    import torch, cv2
    torchscript_model = YOLO("runs/detect/train/weights/best.torchscript")
    results = torchscript_model(img_path)
    print(results)

if __name__ == "__main__":
    # pytorch_demo()
    torchscript_demo()
