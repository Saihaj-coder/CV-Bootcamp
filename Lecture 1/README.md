# Lecture 1 — OpenCV Basics & Motion Detection

Introduction to OpenCV: reading images, basic processing, drawing annotations, and simple motion detection from video.

## Projects

| Script | Description |
|--------|-------------|
| `read_image.py` | Load, display, and save a copy of an image |
| `image_processing.py` | Resize, grayscale, blur, and Canny edge detection |
| `annotation.py` | Draw lines, shapes, and text on a blank canvas |
| `burgler_presence.py` | Detect motion in a video using frame differencing |

## Run

From the repo root (with your virtual environment activated):

```powershell
python "Lecture 1/read_image.py"
python "Lecture 1/image_processing.py"
python "Lecture 1/annotation.py"
python "Lecture 1/burgler_presence.py"
```

Or from this folder:

```powershell
cd "Lecture 1"
python burgler_presence.py
```

Press **ESC** to exit the motion detection video window.

## Assets

- `assets/cmu.jpg` — sample image for image I/O and processing scripts
- `assets/video.mp4` — sample video for motion detection

## Key concepts

- `cv2.imread` / `cv2.imshow` / `cv2.imwrite`
- Color conversion, resize, Gaussian blur, Canny edges
- Drawing primitives (`line`, `rectangle`, `circle`, `ellipse`, `putText`)
- Frame differencing, thresholding, contours, bounding boxes

## Output

Generated files are saved to `output/` (e.g. `cmu_copy.jpg`, `motion_frame_*.jpg`) and are not tracked in git.
