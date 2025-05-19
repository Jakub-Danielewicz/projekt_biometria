import cv2
import numpy as np
import os   

img = cv2.imread("python_model/data/processed_images/table_1.png") #idk czy to tylko mój problem ale inaczej mi nie działało

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()