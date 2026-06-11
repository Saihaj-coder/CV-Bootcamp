from pathlib import Path

import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "yolov8n.pt"

cap = cv2.VideoCapture(str(BASE_DIR / "assets" / "video_2.mp4"))

model = YOLO(str(MODEL_PATH))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=[0])  # class 0 = person

    annotated_frame = results[0].plot()

    cv2.imshow("Annotated video", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
