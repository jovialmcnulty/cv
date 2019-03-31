import numpy as np
import cv2
from cv2 import aruco

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
print(aruco_dict)
# second parameter is id number
# last parameter is total image size
img = aruco.drawMarker(aruco_dict, 5, 700)
cv2.imwrite("marker.jpg", img)
