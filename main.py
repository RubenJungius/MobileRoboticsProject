import cv2
import cv2.aruco as aruco
from skimage.filters import threshold_otsu
import numpy as np
import matplotlib.pyplot as plt
from Vision.vison_functions import*
#from Vision.find_obstacle_slider import*
############################ global var ###############################


# ArUco dictionary
dict_id = cv2.aruco.DICT_6X6_50
arucoDict = cv2.aruco.getPredefinedDictionary(dict_id)
arucoParams = cv2.aruco.DetectorParameters()

corner_ids = {
    1:0,
    2:1,
    3:2,
    4:3
}

# Ids
thymio_id = 5


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
#create_sliders()
print("\nMap computing")
ret, frame = cam.read()
map = crop_labyrinth(frame, M)
cv2.imwrite("Global_map.png", map)

print("\nFind Thymio")
thymio_pos = None
initial_pos= None

#iterate over the few first frame to look for thymio (id=5)
while initial_pos is None:
    ret, frame = cam.read()
    dst = crop_labyrinth(frame, M)
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    detected = cv2.aruco.detectMarkers(dst_gray, arucoDict, parameters=arucoParams)
    (thymio_corners, initial_pos, _) = find_thymio(detected)

#fill the thymio aruco with white
aruco_fill(map, thymio_corners)

#find the obstacle by thresholding
obstacle_map = find_obstacle(map)
cv2.imwrite("Obstacle_map.png",obstacle_map)

end_point=find_end_point(map)
# Check if end_point and thymio_pos are not None before drawing
if end_point is not None:
    # Convert tuple to integers directly
    end_point_int = (int(end_point[0]), int(end_point[1]))
    # Check if the end_point is within the bounds of the map
    if 0 <= end_point_int[0] < obstacle_map.shape[1] and 0 <= end_point_int[1] < obstacle_map.shape[0]:
        cv2.circle(obstacle_map, end_point_int, 5, (0, 0, 255), -1)
    else:
        print("End point is outside the visible region of the map.")
print(end_point)

if initial_pos is not None:
    # Convert tuple to integers directly
    thymio_pos_int = (int(initial_pos[0]), int(initial_pos[1]))
    cv2.circle(obstacle_map, thymio_pos_int, 5, (0, 0, 255), -1)
# Save the image
cv2.imwrite("Obstacle_map+pos.png", obstacle_map)


#################################### main vision algo ########################################

while True:
    ret, frame = cam.read()
    dst = crop_labyrinth(frame, M)
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    detected = cv2.aruco.detectMarkers(dst_gray, arucoDict, parameters=arucoParams)
    thymio_pos = find_thymio(detected)
    draw_thymio(dst, thymio_pos)
    # Display the frame with detected markers and Thymio
    cv2.imshow('Frame with Detected Markers and Thymio', dst)


    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()