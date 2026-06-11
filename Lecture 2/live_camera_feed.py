import cv2
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent.parent / "yolov8n.pt"

cap = cv2.VideoCapture(0)

model = YOLO(str(MODEL_PATH))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("Live Camera Feed", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
