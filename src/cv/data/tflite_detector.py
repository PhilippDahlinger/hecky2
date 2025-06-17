import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

from typing import List, Tuple


CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5


class BoundingBox:
    def __init__(self, x1, y1, x2, y2, cx, cy, w, h, cnf, cls, cls_name):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.cnf = cnf
        self.cls = cls
        self.cls_name = cls_name

    def __repr__(self):
        return f"Box(cls={self.cls_name}, conf={self.cnf:.2f}, [{self.x1:.2f}, {self.y1:.2f}, {self.x2:.2f}, {self.y2:.2f}])"


class Detector:
    def __init__(self, model_path: str, label_path: str):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.tensor_width = self.input_details[0]['shape'][1]
        self.tensor_height = self.input_details[0]['shape'][2]

        with open(label_path, 'r') as f:
            self.labels = [line.strip() for line in f if line.strip()]

    def detect(self, image: Image.Image) -> List[BoundingBox]:
        img = image.convert("RGB").resize((self.tensor_width, self.tensor_height))
        input_data = np.asarray(img, dtype=np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]  # shape: (num_channels, num_elements)
        return self._best_boxes(output)

    def _best_boxes(self, output: np.ndarray) -> List[BoundingBox]:
        num_channels, num_elements = output.shape
        boxes = []

        for c in range(num_elements):
            max_conf = -1.0
            max_idx = -1

            for j in range(4, num_channels):
                conf = output[j][c]
                if conf > max_conf:
                    max_conf = conf
                    max_idx = j - 4

            if max_conf > CONFIDENCE_THRESHOLD:
                cx = output[0][c]
                cy = output[1][c]
                w = output[2][c]
                h = output[3][c]
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                if not all(0.0 <= v <= 1.0 for v in [x1, y1, x2, y2]):
                    continue

                label = self.labels[max_idx] if max_idx < len(self.labels) else str(max_idx)

                boxes.append(BoundingBox(x1, y1, x2, y2, cx, cy, w, h, max_conf, max_idx, label))

        if not boxes:
            return []

        return self._apply_nms(boxes)

    def _apply_nms(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        boxes = sorted(boxes, key=lambda b: b.cnf, reverse=True)
        selected = []

        while boxes:
            first = boxes.pop(0)
            selected.append(first)
            boxes = [b for b in boxes if self._iou(first, b) < IOU_THRESHOLD]

        return selected

    @staticmethod
    def _iou(box1: BoundingBox, box2: BoundingBox) -> float:
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)

        inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        box1_area = box1.w * box1.h
        box2_area = box2.w * box2.h

        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0.0 else 0.0


if __name__ == "__main__":
    import cv2
    import numpy as np
    from PIL import Image

    # Paths
    image_path = "/home/philipp/projects/flutter_projects/rescue_hecky/download/PXL_20250515_055941916.jpg_compressed.JPEG"
    model_path = "/home/philipp/projects/flutter_projects/rescue_hecky/extracted_apk/assets/model.tflite"
    label_path = "/home/philipp/projects/flutter_projects/rescue_hecky/extracted_apk/assets/labels.txt"

    # Init detector and read image
    detector = Detector(model_path, label_path)
    image = Image.open(image_path)
    boxes = detector.detect(image)

    # Convert PIL image to OpenCV format (numpy array)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_h, img_w, _ = image_cv.shape

    # Draw boxes
    for box in boxes:
        pt1 = (int(box.x1 * img_w), int(box.y1 * img_h))
        pt2 = (int(box.x2 * img_w), int(box.y2 * img_h))
        cv2.rectangle(image_cv, pt1, pt2, color=(0, 255, 0), thickness=2)

        label = f"{box.cls_name} ({box.cnf:.2f})"
        cv2.putText(image_cv, label, (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(0, 255, 0), thickness=1)

    # Show or save the image
    cv2.imshow("Detections", image_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optional: Save to file
    # cv2.imwrite("output_with_boxes.jpg", image_cv)
