# Computer Vision Bootcamp

Projects and exercises from a hands-on computer vision bootcamp, organized by lecture.

## Setup

```powershell
python -m venv hands_on_cv
.\hands_on_cv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Lectures

| Lecture | Topic | Folder |
|---------|-------|--------|
| 1 | OpenCV basics, image processing, motion detection | [Lecture 1](Lecture%201/) |

## Project structure

```
├── Lecture N/
│   ├── README.md       # What this lecture covers and how to run it
│   ├── assets/         # Input images and videos (committed)
│   ├── output/         # Generated files (gitignored)
│   └── *.py            # Scripts
├── requirements.txt
└── README.md
```

## Notes

- Run scripts from any directory; paths are resolved relative to each script.
- Large or generated outputs are saved to `output/` and excluded from git.
