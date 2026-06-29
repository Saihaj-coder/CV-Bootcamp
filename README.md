# Computer Vision Bootcamp

Projects and exercises from a hands-on computer vision bootcamp, organized by lecture.

## Setup

```powershell
python -m venv hands_on_cv
.\hands_on_cv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Lecture 1 and 2 use Python scripts. Lecture 3 uses Jupyter notebooks (Google Colab or local Jupyter with GPU recommended).

## Lectures

| Lecture | Topic | Folder |
|---------|-------|--------|
| 1 | OpenCV basics, image processing, motion detection | [Lecture 1](Lecture%201/) |
| 2 | YOLO detection, tracking, counting, segmentation | [Lecture 2](Lecture%202/) |
| 3 | R-CNN family — R-CNN, Faster R-CNN, Mask R-CNN | [Lecture 3](Lecture%203/) |

## Project structure

```
├── Lecture N/
│   ├── README.md       # What this lecture covers and how to run it
│   ├── assets/         # Input images and videos (where applicable)
│   ├── output/         # Generated files (gitignored)
│   ├── *.py            # Scripts (Lectures 1–2)
│   └── *.ipynb         # Notebooks (Lecture 3+)
├── requirements.txt
└── README.md
```

## Notes

- Run scripts from any directory; paths are resolved relative to each script.
- YOLO model weights (`yolov8n.pt`, `yolov8n-seg.pt`) auto-download on first run and are excluded from git.
- Large or generated outputs are saved to `output/` and excluded from git — re-run the scripts to recreate them.
- Lecture 3 notebooks may download datasets and checkpoints at runtime; see [Lecture 3 README](Lecture%203/) for details.
