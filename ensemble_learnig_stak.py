import cv2
import numpy as np
from ultralytics import YOLO

# Load five YOLOv8 models
model_paths = [
    r"C:\Users\ayuba\Downloads\best (9).pt",
    r"C:\Users\ayuba\Downloads\best (10).pt",
    r"C:\Users\ayuba\Downloads\best (13).pt",
    r"C:\Users\ayuba\Downloads\best (12).pt",
    r"C:\Users\ayuba\Downloads\best (11).pt"
]
models = [YOLO(path) for path in model_paths]

# --------------------------- ENSEMBLE METHODS ---------------------------

# 1. Stacking: Weighted model confidence (favor best model)
def run_stacking_inference(frame):
    model_weights = [0.9, 0.9, 0.9, 0.9, 1.2]
    all_preds = []

    for i, model in enumerate(models):
        result = model(frame, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy()

        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            all_preds.append([*box, conf * model_weights[i], int(cls_id)])

    all_preds = np.array(all_preds)
    if len(all_preds) == 0:
        return []
    return non_max_suppression(all_preds, iou_threshold=0.5)

# 2. Bagging: Equal vote from all models
def run_bagging_inference(frame):
    all_preds = []

    for model in models:
        result = model(frame, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy()

        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            all_preds.append([*box, conf, int(cls_id)])

    all_preds = np.array(all_preds)
    if len(all_preds) == 0:
        return []
    return non_max_suppression(all_preds, iou_threshold=0.5)

# 3. Boosting: Later models have higher confidence weight
def run_boosting_inference(frame):
    boost_weights = [0.7, 0.9, 1.0, 1.1, 1.3]
    all_preds = []

    for i, model in enumerate(models):
        result = model(frame, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy()

        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            all_preds.append([*box, conf * boost_weights[i], int(cls_id)])

    all_preds = np.array(all_preds)
    if len(all_preds) == 0:
        return []
    return non_max_suppression(all_preds, iou_threshold=0.5)

# --------------------------- NMS UTILITY ---------------------------
def non_max_suppression(preds, iou_threshold=0.5):
    boxes = preds[:, :4]
    scores = preds[:, 4]
    classes = preds[:, 5].astype(int)

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=scores.tolist(),
        score_threshold=0.3,
        nms_threshold=iou_threshold
    )

    final_results = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_results.append({
                'box': boxes[i],
                'score': scores[i],
                'class': classes[i]
            })

    return final_results

# --------------------------- VIDEO INFERENCE ---------------------------
video_path = r"C:\Users\ayuba\Downloads\WLCCTVNVR_ch6_main_20250707212400_20250707235959.mp4"
cap = cv2.VideoCapture(video_path)

output_path = 'ensemble_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ------------------ Choose One Inference Method ------------------
    # results = run_stacking_inference(frame)   # ✅ Stacking
    results = run_bagging_inference(frame)   # ✅ Bagging
    # results = run_boosting_inference(frame)  # ✅ Boosting

    # Draw results
    for res in results:
        x1, y1, x2, y2 = map(int, res['box'])
        label = f"Class {res['class']} {res['score']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    out.write(frame)
    cv2.imshow('YOLOv8 Ensemble (Bagging/Boosting/Stacking)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
