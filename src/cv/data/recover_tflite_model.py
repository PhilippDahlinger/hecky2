import numpy as np
import cv2
from PIL import Image
import tflite_runtime.interpreter as tflite


def calculate_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection_area / (box1_area + box2_area - intersection_area + 1e-6)
    return iou

def apply_nms(boxes, scores, iou_threshold=IOU_THRESHOLD):
    indices = np.argsort(scores)[::-1]  # sort by confidence descending
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        if len(indices) == 1:
            break

        rest = indices[1:]
        ious = np.array([calculate_iou(boxes[current], boxes[i]) for i in rest])
        indices = rest[ious < iou_threshold]

    return keep

def extract_boxes(output, labels):
    # output shape: (1, num_channels, num_elements)
    output = output[0]  # remove batch dimension
    num_channels, num_elements = output.shape

    bounding_boxes = []
    confidences = []
    classes = []

    for c in range(num_elements):
        # Extract box coordinates (cx, cy, w, h)
        cx = output[0, c]
        cy = output[1, c]
        w = output[2, c]
        h = output[3, c]

        # Extract class confidences (channels 4 to end)
        class_scores = output[4:, c]
        max_conf = np.max(class_scores)
        max_class_idx = np.argmax(class_scores)

        if max_conf > CONFIDENCE_THRESHOLD:
            # Convert to x1,y1,x2,y2 (normalized coordinates)
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            # Skip boxes with invalid coordinates
            if not (0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1):
                continue

            bounding_boxes.append([x1, y1, x2, y2])
            confidences.append(max_conf)
            classes.append(max_class_idx)

    if len(bounding_boxes) == 0:
        return []

    bounding_boxes = np.array(bounding_boxes)
    confidences = np.array(confidences)
    classes = np.array(classes)

    keep_indices = apply_nms(bounding_boxes, confidences, IOU_THRESHOLD)

    results = []
    for i in keep_indices:
        results.append({
            'bbox': bounding_boxes[i].tolist(),
            'confidence': confidences[i],
            'class_id': int(classes[i]),
            'class_name': labels[int(classes[i])]
        })

    return results

# Example usage:
# output = ...  # your numpy array from interpreter.get_tensor(output_details[0]['index'])
# labels = [...]  # list of class label strings loaded from your labels.txt file
# boxes = extract_boxes(output, labels)
# print(boxes)

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="/home/philipp/projects/flutter_projects/rescue_hecky/extracted_apk/assets/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Check input shape
input_shape = input_details[0]['shape']  # e.g. [1, 640, 640, 3]

# Load and preprocess an image
img = Image.open("/home/philipp/projects/flutter_projects/rescue_hecky/download/PXL_20250515_055941916.jpg_compressed.JPEG").convert("RGB").resize((640, 640))

# Convert to numpy float32 array and normalize
input_data = np.array(img, dtype=np.float32) / 255.0

# Add batch dimension
input_data = np.expand_dims(input_data, axis=0)  # Shape: (1, 640, 640, 3)

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get results
output_data = interpreter.get_tensor(output_details[0]['index'])
output = output_data  # (1, 5, 8400)


CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
