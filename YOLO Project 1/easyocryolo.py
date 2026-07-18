import re
from collections import defaultdict, deque
from pathlib import Path

import cv2
import easyocr
import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "license_plate_best.pt"
INPUT_VIDEO = BASE_DIR / "license_plate_detection_input.mp4"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
# MJPG in AVI plays reliably on Windows; mp4v often produces files players reject
OUTPUT_VIDEO = OUTPUT_DIR / "output_with_license.avi"

model = YOLO(str(MODEL_PATH))
reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

# UK-style format: 2 letters + 2 numbers + 3 letters (e.g. AB12CDE)
plate_pattern = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$")


def correct_plate_format(ocr_text):
    mapping_num_to_alpha = {"0": "O", "1": "I", "5": "S", "8": "B"}
    mapping_alpha_to_num = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8"}

    ocr_text = ocr_text.upper().replace(" ", "")
    if len(ocr_text) != 7:
        return ""

    corrected = []
    for i, ch in enumerate(ocr_text):
        if i < 2 or i >= 4:
            if ch.isdigit() and ch in mapping_num_to_alpha:
                corrected.append(mapping_num_to_alpha[ch])
            elif ch.isalpha():
                corrected.append(ch)
            else:
                return ""
        else:
            if ch.isalpha() and ch in mapping_alpha_to_num:
                corrected.append(mapping_alpha_to_num[ch])
            elif ch.isdigit():
                corrected.append(ch)
            else:
                return ""

    return "".join(corrected)


def recognize_plate(plate_crop):
    if plate_crop.size == 0:
        return ""

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plate_resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    try:
        ocr_result = reader.readtext(
            plate_resized,
            detail=0,
            allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        )
        if ocr_result:
            raw_text = "".join(ocr_result).upper()
            candidate = correct_plate_format(raw_text)
            if candidate and plate_pattern.match(candidate):
                return candidate
    except Exception:
        pass

    return ""


plate_history = defaultdict(lambda: deque(maxlen=10))
plate_final = {}


def get_box_id(x1, y1, x2, y2):
    return f"{int(x1 / 10)}_{int(y1 / 10)}_{int(x2 / 10)}_{int(y2 / 10)}"


def get_stable_plate(box_id, new_text):
    if new_text:
        plate_history[box_id].append(new_text)
        most_common = max(set(plate_history[box_id]), key=plate_history[box_id].count)
        plate_final[box_id] = most_common
    return plate_final.get(box_id, "")


cap = cv2.VideoCapture(str(INPUT_VIDEO))
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if width <= 0 or height <= 0:
    cap.release()
    raise RuntimeError("Could not read video dimensions from input file.")

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (width, height))
if not out.isOpened():
    cap.release()
    raise RuntimeError(f"Could not create output video: {OUTPUT_VIDEO}")

CONF_THRESH = 0.3
frame_count = 0
SHOW_PREVIEW = True

try:
    cv2.namedWindow("Annotated Video", cv2.WINDOW_NORMAL)
except cv2.error:
    SHOW_PREVIEW = False
    print("Live preview unavailable (headless OpenCV). Saving video only.")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                if conf < CONF_THRESH:
                    continue

                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                plate_crop = frame[y1:y2, x1:x2]

                text = recognize_plate(plate_crop)

                box_id = get_box_id(x1, y1, x2, y2)
                stable_text = get_stable_plate(box_id, text)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                if plate_crop.size > 0:
                    overlay_h, overlay_w = 150, 400
                    plate_overlay = cv2.resize(plate_crop, (overlay_w, overlay_h))

                    oy1 = max(0, y1 - overlay_h - 40)
                    ox1 = x1
                    oy2, ox2 = oy1 + overlay_h, ox1 + overlay_w

                    if oy2 <= frame.shape[0] and ox2 <= frame.shape[1]:
                        frame[oy1:oy2, ox1:ox2] = plate_overlay

                        if stable_text:
                            cv2.putText(frame, stable_text, (ox1, oy1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6)
                            cv2.putText(frame, stable_text, (ox1, oy1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        out.write(frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

        if SHOW_PREVIEW:
            cv2.imshow("Annotated Video", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped early by user (q pressed).")
                break
finally:
    cap.release()
    out.release()
    if SHOW_PREVIEW:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

if frame_count == 0:
    raise RuntimeError("No frames were written. Output video would be empty.")

print(f"Saved {frame_count} frames to {OUTPUT_VIDEO}")
print("Open with VLC or Movies & TV if your default player has issues.")
