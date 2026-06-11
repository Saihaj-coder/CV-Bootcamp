from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "yolov8n.pt"

model = YOLO(str(MODEL_PATH))
cap = cv2.VideoCapture(str(BASE_DIR / "assets" / "cars2.mp4"))
unique_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame, classes=[2], persist=True, verbose=False)  # classes=[2] = car
    annotated_frame = results[0].plot()

    if results[0].boxes and results[0].boxes.id is not None:
        ids = results[0].boxes.id.numpy()
        for oid in ids:
            unique_ids.add(oid)
        cv2.putText(annotated_frame, f"Count: {len(unique_ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Object Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()