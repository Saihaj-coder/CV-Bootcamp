from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "yolov8n.pt"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "people_with_trail.mp4"

model = YOLO(str(MODEL_PATH))
cap = cv2.VideoCapture(str(BASE_DIR / "assets" / "video_2.mp4"))

fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

id_map = {}
next_id = 1

trail = defaultdict(lambda: deque(maxlen=30))
appear = defaultdict(int)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    res = model.track(frame, classes=[0], persist=True, verbose=False)
    annotated_frame = frame.copy()

    if res[0].boxes.id is not None:
        boxes = res[0].boxes.xyxy.numpy()
        ids = res[0].boxes.id.numpy()

        for box, oid in zip(boxes, ids):
            oid = int(oid)
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            appear[oid] += 1

            if appear[oid] >= 5 and oid not in id_map:
                id_map[oid] = next_id
                next_id += 1

            if oid in id_map:
                sid = id_map[oid]
                trail[oid].append((cx, cy))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"ID: {sid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

    for points in trail.values():
        if len(points) > 1:
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [pts], False, (0, 255, 255), 2)

    writer.write(annotated_frame)

    cv2.imshow("Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"Saved annotated video to {OUTPUT_PATH}")
