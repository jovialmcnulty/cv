import numpy as np
import cv2
import cv2.aruco as aruco
import sys
import yaml

#read an image file, detect it's aruco codes, and draw them onto an output image

with open('calibration.yaml') as f:
    loadeddict = yaml.load(f)

camera_matrix = np.asarray(loadeddict.get('camera_matrix'))
dist_coeff = np.asarray(loadeddict.get('dist_coeff'))

print(camera_matrix)
print(dist_coeff)

marker_size = .01905 #meters
in_file = sys.argv[1] + '.jpg'
out_file = sys.argv[1] + '_tested.jpg'
im = cv2.imread(in_file)

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
print(corners)
aruco.drawDetectedMarkers(im, corners, ids)

rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeff)
print(rvecs)
print(tvecs)
for i in range(len(rvecs)) :
    aruco.drawAxis(im, camera_matrix, dist_coeff, rvecs[i], tvecs[i], marker_size*2)

cv2.imwrite(out_file, im)
