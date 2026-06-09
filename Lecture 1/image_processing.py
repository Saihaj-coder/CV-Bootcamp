from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent
img = cv2.imread(str(BASE_DIR / "assets" / "cmu.jpg"))

resizing = cv2.resize(img, (640, 480))
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurring = cv2.GaussianBlur(img, (5, 5), 0)
edges = cv2.Canny(img, 100, 200)

cv2.imshow("Resized image", resizing)
cv2.imshow("Grey image", grey)
cv2.imshow("Blurred image", blurring)
cv2.imshow("Edges image", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
