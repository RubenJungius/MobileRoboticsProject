import cv2
import cv2.aruco as aruco
from skimage.filters import threshold_otsu
import numpy as np
import matplotlib.pyplot as plt
from Vision.vison_functions import*
import time


############################ Code ###############################

# Open Camera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Unable to open the camera.")
else:
    print("\nSearching corners and correct perspective")

cam.set(cv2.CAP_PROP_FRAME_WIDTH, res_h)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, res_w)

#search for the 4 corner and the right transformation M
M=corners_calibration(cam)

print("\nMap computing")

time.sleep(2)

ret, frame = cam.read()
map = get_ROI(frame, M)
cv2.imwrite("Global_map.png", map)

print("\nFind Thymio")
#initialise the position
thymio_pos = None
initial_pos= None
end_point= None

#iterate over the few first frame to look for thymio (id=5)
while initial_pos is None or end_point is None:
    ret, frame = cam.read()
    dst = get_ROI(frame, M)
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    detected = cv2.aruco.detectMarkers(dst_gray, arucoDict, parameters=arucoParams)
    (thymio_corners, initial_pos, _) = find_thymio(detected)
    (end_point_corners,end_point)=find_end_point(detected)

#fill the thymio aruco with white -> thymio+end_point
aruco_fill(map, thymio_corners)
aruco_fill(map, end_point_corners)


#find the obstacle by thresholding
obstacle_map = find_obstacle(map)
cv2.imwrite("Obstacle_map_filtered.png",obstacle_map)

#map the obstacle, the thymio and the end point in the same map
draw_final_map(initial_pos,end_point,obstacle_map)

#################################### main vision algo ########################################

while True:
    ret, frame = cam.read()
    roi = get_ROI(frame, M)
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) #gray transform for aruco detection
    detected = cv2.aruco.detectMarkers(roi_gray, arucoDict, parameters=arucoParams)
    thymio_pos = find_thymio(detected)
    draw_thymio(dst, thymio_pos)
    # Display the frame with detected markers and Thymio
    cv2.imshow('Frame with Detected Markers and Thymio', roi)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()