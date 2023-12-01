import cv2
import cv2.aruco as aruco
from skimage.filters import threshold_otsu
import numpy as np
import matplotlib.pyplot as plt

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
  # blacks squares have to be to te front of the thymio
  v1 = c[1] - c[0]
  v2 = c[2] - c[3]
  dir = (v1 + v2)/2
  angle = np.arctan2(dir[1], dir[0])
  
  return (c, center, angle)


def draw_thymio(output_image, thymio_pos):
    c, center, angle = thymio_pos
    if center is not None:

        # Draw red dot at the center
        cv2.circle(output_image, tuple(center.astype(int)), 5, (0, 0, 255), -1)

        # Draw arrow indicating direction
        length = 30
        arrow_tip = (
            int(center[0] + length * np.cos(angle)),
            int(center[1] + length * np.sin(angle))
        )
        cv2.arrowedLine(output_image, tuple(center.astype(int)), arrow_tip, (0, 0, 255), 2)


def binarisation(im):
    red_channel = im[:, :, 0]
    threshold = threshold_otsu(red_channel)
    binary = np.asarray((im > threshold).astype(float))
    binary[binary==1] = 255
    return binary


def find_obstacle(image):
    #binary = binarisation(image)
    #gray = cv2.cvtColor(binary.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    #contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #obstacle_map = np.zeros_like(gray, dtype=np.uint8)
    cv2.imwrite("ch1.png", image)
    gray_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("ch15.png", gray_im)
    #threshold = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    threshold = 100
    threshold_value, obstacle_map = cv2.threshold(gray_im, threshold, 255, cv2.THRESH_BINARY)
    #gray = cv2.cvtColor(binary.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(obstacle_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.imwrite("ch2.png", obstacle_map)

    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area < 20000:
            # Draw the filled contour on the obstacle map
            cv2.drawContours(obstacle_map, [cnt], -1, 255, thickness=cv2.FILLED)


    cv2.imwrite("ch3.png", obstacle_map)
    height, width = obstacle_map.shape[:2]

    # size of the black spot corner 
    width_corners = 60

    # #erase tags on the corners
    obstacle_map[:width_corners, :width_corners] = 255  # Coin supérieur gauche
    obstacle_map[:width_corners, width - width_corners:] = 255  # Coin supérieur droit
    obstacle_map[height - width_corners:, :width_corners] = 255  # Coin inférieur gauche
    obstacle_map[height - width_corners:, width - width_corners:] = 255  # Coin inférieur droit

    obstacle_map = cv2.bitwise_not(obstacle_map)
    cv2.imwrite("ch4.png", obstacle_map)
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


def aruco_fill(frame, corners):
    if corners is not None:
        # Reshape thymio_corners to (4, 1, 2)
        c = np.int32(corners.reshape((4, 1, 2)))

        # Fill the shape with white
        cv2.fillPoly(frame, [c], color=(255, 255, 255))
        return
    
def find_end_point(image):
    blue_channel = image[:, :, 2]
    low_threshold = 110  # to_tune
    high_threshold = 150 #to_tune
    blue_part = ((blue_channel > low_threshold) & (blue_channel < high_threshold)).astype(np.uint8) * 255

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bri_channel= hsv_image[:, :, 2]
    threshold_value = 100  # to_tune
    _, white_part = cv2.threshold(bri_channel, threshold_value, 255, cv2.THRESH_BINARY)

    binary = cv2.bitwise_xor(blue_part, white_part)

    cv2.imwrite("buffer.png", binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours based on area, you may need to adjust this threshold
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 600]

    # Check if any contours are found
    if filtered_contours:
        # Choose the first contour (you may need to refine the selection)
        contour = filtered_contours[0]

        # Calculate the moments of the contour
        moments = cv2.moments(contour)

        # Avoid division by zero
        if moments['m00'] != 0:
            # Calculate the centroid
            centroid_x = int(moments['m10'] / moments['m00'])
            centroid_y = int(moments['m01'] / moments['m00'])
            return centroid_x, centroid_y

    # Return None if no valid contour is found
    return None
