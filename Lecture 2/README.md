# Lecture 2 — YOLO Object Detection, Tracking & Segmentation

Introduction to Ultralytics YOLO: detection on images and video, live webcam inference, multi-object tracking, counting, motion trails, and instance segmentation.

## Projects

| Script | Description |
|--------|-------------|
| `simple_object_detection.py` | Detect people in a still image (`man.jpg`) |
| `multi_object_from_video.py` | Person detection on a video file |
| `live_camera_feed.py` | Real-time object detection from webcam |
| `object_counting.py` | Track cars and count unique vehicle IDs over a video |
| `people_with_trail.py` | Track people and draw movement trails; saves annotated video |
| `segmentation.py` | Person segmentation with contour outlines and track IDs; saves annotated video |

## Run

From the repo root (with your virtual environment activated):

```powershell
python "Lecture 2/simple_object_detection.py"
python "Lecture 2/multi_object_from_video.py"
python "Lecture 2/live_camera_feed.py"
python "Lecture 2/object_counting.py"
python "Lecture 2/people_with_trail.py"
python "Lecture 2/segmentation.py"
```

Press **q** to quit video/webcam windows.

On first run, YOLO downloads pretrained weights to the project root (`yolov8n.pt`, `yolov8n-seg.pt`).

## Assets

| File | Used by |
|------|---------|
| `assets/man.jpg` | `simple_object_detection.py` |
| `assets/video_2.mp4` | `multi_object_from_video.py`, `people_with_trail.py` |
| `assets/cars2.mp4` | `object_counting.py` |
| `assets/people_walking.mp4` | `segmentation.py` |

## Key concepts

- YOLO pretrained checkpoints (`.pt`) and COCO class IDs
- `model(image)` vs `model.track(..., persist=True)` for tracking
- Filtering classes (e.g. `0` = person, `2` = car)
- Unique ID counting with `boxes.id`
- Segmentation masks, contours, and manual annotation
- Saving annotated output with `cv2.VideoWriter`

## Output

Scripts that save video write to `output/`:

| Script | Output file |
|--------|-------------|
| `people_with_trail.py` | `output/people_with_trail.mp4` |
| `segmentation.py` | `output/segmentation.mp4` |

Generated outputs are **not** tracked in git (regenerated when you run the scripts). See the root README for the recommended workflow.
