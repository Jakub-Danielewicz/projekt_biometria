import cv2
import numpy as np
import os   

img = cv2.imread("data/processed_images/table_1.png")

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()