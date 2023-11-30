import cv2
import cv2.aruco as aruco
from skimage.filters import threshold_otsu
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pyvisgraph as vg


from timeit import default_timer as timer
from tdmclient import ClientAsync, aw
from src.FollowPath import*
from src.Obstacle_avoid import*
from Vision.vison_functions import*
from src.globalNavigation import*

############################ global var ###############################


# ArUco dictionary
dict_id = cv2.aruco.DICT_6X6_50
arucoDict = cv2.aruco.getPredefinedDictionary(dict_id)
arucoParams = cv2.aruco.DetectorParameters()

## A0 paper ratio --> a mesurer pour avoir les distances entre les points tu coco
res_w = 720
res_h = 1020

corner_ids = {
    1:0,
    2:1,
    3:2,
    4:3
}

# Ids
thymio_id = 5

 
############################ function def ###############################

# Given perspective transform, crops the original image
def crop_labyrinth(img, M):
    return cv2.warpPerspective(img,M,(res_h,res_w))


def corners_calibration(cam): #load
    while True:
        ret, frame = cam.read()

        M=perspective_correction(frame)
        if M is not None:
            return M
        
        print("no corner detected")
        return


def get_pos_aruco(detected, search_id):
  (corners, ids, rejected) = detected

  if ids is not None:
    for i, id in enumerate(ids):
      c = corners[i][0]
      if id[0] == search_id:
        center = (c[0]+c[1]+c[2]+c[3])/4
        return (center, c)
  return (None, None)


def find_thymio(detected):
  # Detect aruco
  (center, c) = get_pos_aruco(detected, thymio_id)

  if center is None:
    return (None, None, None)

  # c[0]        TOP LEFT
  # c[1]        BOTTOM RIGHT
  # c[2]        BOTTOM LEFT
  # c[3]        TOP LEFT
  # Compute orientation
  top_middle = (c[0]+c[3])/2
  dir = top_middle - center
  angle = np.arctan2(dir[0], dir[1])
  
  return (c, center, angle)


def draw_thymio(output_image, thymio_pos):
    if thymio_pos is not None:
        c, center, angle = thymio_pos

        # Draw red dot at the center
        cv2.circle(output_image, tuple(center.astype(int)), 5, (0, 0, 255), -1)

        # Draw arrow indicating direction
        length = 30
        arrow_tip = (
            int(center[0] + length * np.sin(angle)),
            int(center[1] - length * np.cos(angle))
        )
        cv2.arrowedLine(output_image, tuple(center.astype(int)), arrow_tip, (0, 0, 255), 2)


def binarisation(im):
    green_channel = im[:, :, 0]
    threshold = threshold_otsu(green_channel)
    binary = np.asarray((im < threshold).astype(float))
    binary[binary==1] = 255
    return binary



def find_obstacle(image):
    binary = binarisation(image)
    gray = cv2.cvtColor(binary.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    obstacle_map = np.zeros_like(gray, dtype=np.uint8)

    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)

        if area > 4000:
            # Draw the filled contour on the obstacle map
            cv2.drawContours(obstacle_map, [cnt], -1, 255, thickness=cv2.FILLED)

    return obstacle_map


def perspective_correction(image):
    detected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
    cs = [None]*4
    for (id, idx) in corner_ids.items():
      (center, _) = get_pos_aruco(detected, id)
      if center is not None:
        cs[idx] = center
      else:
        return None

    # Do perspective correction
    pts1 = np.array([cs[0], cs[1], cs[3], cs[2]])
    pts2 = np.float32([[1020,720], [0, 720], [1020, 0], [0, 0]])

    return cv2.getPerspectiveTransform(pts1,pts2)

############################ code ###############################


# Open Camera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Unable to open the camera.")
else:
    print("\nSearching corners and correct perspective")

cam.set(cv2.CAP_PROP_FRAME_WIDTH, res_h)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, res_w)

M=corners_calibration(cam)


print("\nMap computing")
ret, frame = cam.read()
cv2.imshow("sdf",frame)
map = crop_labyrinth(frame, M)
cv2.imwrite("Global_map.png", map)

obstacle_map = find_obstacle(map)
cv2.imwrite("Obstacle_map.png",obstacle_map)

print("\nFind Thymio")
thymio_pos = None
initial_pos= None

while initial_pos is None:
    ret, frame = cam.read()
    dst = crop_labyrinth(frame, M)
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    detected = cv2.aruco.detectMarkers(dst_gray, arucoDict, parameters=arucoParams)
    (_, initial_pos, _) = find_thymio(detected)



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