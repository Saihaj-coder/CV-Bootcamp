import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "yolov8n-seg.pt"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "segmentation.mp4"

# Refinement: use MODEL_PATH so weights load from the project root (same as other Lecture 2 scripts)
model = YOLO(str(MODEL_PATH))
cap = cv2.VideoCapture(str(BASE_DIR / "assets" / "people_walking.mp4"))

fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

while True:
    ret, frame = cap.read()

    # Refinement: stop when the video ends instead of looping on empty frames
    if not ret:
        break

    results = model.track(source=frame, classes=[0], persist=True, verbose=False)

    # Refinement: start each frame as a copy so we draw on top without mutating the original
    annotated_frame = frame.copy()

    for r in results:
        if r.masks is not None and r.boxes is not None and r.boxes.id is not None:
            masks = r.masks.data.numpy()
            boxes = r.boxes.xyxy.numpy()
            ids = r.boxes.id.numpy()

            for i, mask in enumerate(masks):
                person_id = int(ids[i])
                x1, y1, x2, y2 = map(int, boxes[i])

                # Resize mask to full frame size so contours align with the video
                mask_resized = cv2.resize(
                    mask.astype(np.uint8) * 255,
                    (frame.shape[1], frame.shape[0]),
                )
                contours, _ = cv2.findContours(
                    mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(annotated_frame, contours, -1, (0, 0, 255), 2)
                cv2.putText(
                    annotated_frame,
                    f"ID: {person_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

    writer.write(annotated_frame)

    # Refinement: show every frame at the while-loop level (even when no detections)
    # so the window keeps updating instead of freezing on empty frames
    cv2.imshow("Object Tracking with Segmentation", annotated_frame)

    # Refinement: quit check belongs here so 'q' breaks the main video loop, not just the inner for-loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"Saved annotated video to {OUTPUT_PATH}")
