# Computer Vision Bootcamp

Projects and exercises from a hands-on computer vision bootcamp, organized by lecture and project folders.

## Setup

```powershell
python -m venv hands_on_cv
.\hands_on_cv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Lecture 1 and 2 use Python scripts. Lecture 3 uses Jupyter notebooks (Google Colab or local Jupyter with GPU recommended). Capstone-style folders (e.g. YOLO Project 1) may need extra packages such as `easyocr`.

## Lectures & projects

| Item | Topic | Folder |
|------|-------|--------|
| Lecture 1 | OpenCV basics, image processing, motion detection | [Lecture 1](Lecture%201/) |
| Lecture 2 | YOLO detection, tracking, counting, segmentation | [Lecture 2](Lecture%202/) |
| Lecture 3 | R-CNN family — R-CNN, Faster R-CNN, Mask R-CNN | [Lecture 3](Lecture%203/) |
| YOLO Project 1 | License plate detection (YOLOv8) + EasyOCR | [YOLO Project 1](YOLO%20Project%201/) |

## Project structure

```
├── Lecture N/
│   ├── README.md       # What this lecture covers and how to run it
│   ├── assets/         # Input images and videos (where applicable)
│   ├── output/         # Generated files (gitignored)
│   ├── *.py            # Scripts (Lectures 1–2)
│   └── *.ipynb         # Notebooks (Lecture 3+)
├── YOLO Project N/
│   ├── README.md       # Project overview + theory (e.g. YOLOv8)
│   ├── *.py / *.ipynb  # Inference / training notebooks
│   ├── *.pt            # Fine-tuned weights (local; gitignored)
│   └── output/         # Generated videos (gitignored)
├── requirements.txt
└── README.md
```

## Notes

- Run scripts from any directory; paths are resolved relative to each script.
- YOLO model weights (`yolov8n.pt`, `yolov8n-seg.pt`, project `*.pt` files) are excluded from git — place fine-tuned weights locally as documented in each folder README.
- Large or generated outputs are saved to `output/` and excluded from git — re-run the scripts to recreate them.
- Lecture 3 notebooks may download datasets and checkpoints at runtime; see [Lecture 3 README](Lecture%203/) for details.
- See [YOLO Project 1 README](YOLO%20Project%201/) for YOLOv8 architecture, training/loss details, and the license-plate pipeline.
