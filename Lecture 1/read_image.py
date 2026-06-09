from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
OUTPUT_DIR = BASE_DIR / "output"

img1 = cv2.imread(str(ASSETS_DIR / "cmu.jpg"))
cv2.imshow("The original image", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

OUTPUT_DIR.mkdir(exist_ok=True)
cv2.imwrite(str(OUTPUT_DIR / "cmu_copy.jpg"), img1)
