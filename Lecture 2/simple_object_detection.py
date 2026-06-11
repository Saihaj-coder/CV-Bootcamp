from pathlib import Path

import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "yolov8n.pt"

model = YOLO(str(MODEL_PATH))

image = cv2.imread(str(BASE_DIR / "assets" / "man.jpg"))
results = model(image)

annotated_image = results[0].plot()

cv2.imshow("Annotated Image", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
