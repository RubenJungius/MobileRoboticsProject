import cv2
import cv2.aruco as aruco
from skimage.filters import threshold_otsu
import numpy as np
import matplotlib.pyplot as plt
from Vision.find_obstacle_slider import*
############################ vision var ###############################


# ArUco dictionary
dict_id = cv2.aruco.DICT_6X6_50
arucoDict = cv2.aruco.getPredefinedDictionary(dict_id)
arucoParams = cv2.aruco.DetectorParameters()

## cam settings : 720p -> enough resolution/less latency
res_w = 720
res_h = 1280

corner_ids = {
    1:0,
    2:1,
    3:2,
    4:3
}

# Ids
thymio_id = 5
end_point_id = 8

############################ function def ###############################

# Given perspective transform, crops the original image (region of interest)
def get_ROI(img, M): 
    return cv2.warpPerspective(img,M,(res_h,res_w))

#transition function to test if M is well computed
def corners_calibration(cam): 
    while True:
        ret, frame = cam.read()

        M=perspective_correction(frame)
        if M is not None:
            return M
        
        print("no corner detected")
        return


def get_pos_aruco(detected, search_id):
  #return the center and the 4 corners of the aruco
  (corners, ids, rejected) = detected

  if ids is not None:
    for i, id in enumerate(ids):
      corner = corners[i][0]
      if id[0] == search_id:
        center = (corner[0]+corner[1]+corner[2]+corner[3])/4
        return (center, corner)
  return (None, None)


def find_thymio(detected):
  # Detect aruco
  (center, c) = get_pos_aruco(detected, thymio_id)

  if center is None:
    return (None, None, None)

  #compute the direction : 
  v1 = c[1] - c[0]
  v2 = c[2] - c[3]
  dir = (v1 + v2)/2
  angle = np.arctan2(dir[1], dir[0])
  
  return (c, center, angle)


def draw_thymio(output_image, thymio_pos):
    #draw the position and the direction
    corners, center, angle = thymio_pos
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


def find_obstacle(image):
    #use of slider to fine tune the research
    red_binarisation(image)
    gray = cv2.cvtColor(cv2.imread('Obstacle_map.png').astype(np.uint8), cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    obstacle_map = np.zeros_like(gray, dtype=np.uint8)

    #filtering of the small elements to remove noise
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:
            # Draw the filled contour on the obstacle map
            cv2.drawContours(obstacle_map, [cnt], -1, 255, thickness=cv2.FILLED)
    obstacle_map = pre_processing (obstacle_map)
    
    return obstacle_map


def perspective_correction(image):
    detected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
    corners = [None]*4
    for (id, idx) in corner_ids.items():
      (center, _) = get_pos_aruco(detected, id)
      if center is not None:
        corners[idx] = center
      else:
        return None

    # Do perspective correction
    pts1 = np.array([corners[0], corners[1], corners[3], corners[2]])
    pts2 = np.float32([[res_h,res_w], [0, res_w], [res_h, 0], [0, 0]])

    return cv2.getPerspectiveTransform(pts1,pts2)


def aruco_fill(frame, corners):
    if corners is not None:
        # Reshape thymio_corners to be able to use cv2
        c = np.int32(corners.reshape((4, 1, 2)))

        # Fill the shape with white to avoid obstacle detection on the thymio/end_point
        cv2.fillPoly(frame, [c], color=(255, 255, 255))
        return
    

def find_end_point(detected):
  (center, corners) = get_pos_aruco(detected, end_point_id)
  if center is None:
    return None
  return (corners,center)


def draw_final_map(initial_pos,end_point,obstacle_map):
  # Check if end_point and thymio_pos are not None before drawing
  if end_point is not None:
      # Convert tuple to integers directly
      end_point_int = (int(end_point[0]), int(end_point[1]))
      # Check if the end_point is within the bounds of the map
      if 0 <= end_point_int[0] < obstacle_map.shape[1] and 0 <= end_point_int[1] < obstacle_map.shape[0]:
          cv2.circle(obstacle_map, end_point_int, 5, (0, 0, 255), -1)
      else:
          print("End point is outside the visible region of the map.")

  if initial_pos is not None:
      # Convert tuple to integers directly
      thymio_pos_int = (int(initial_pos[0]), int(initial_pos[1]))
      cv2.circle(obstacle_map, thymio_pos_int, 5, (0, 0, 255), -1)
  # Save the image
  cv2.imwrite("Starting_map.png", obstacle_map)
  return

def pre_processing(edge, min_size=4000):
    # Perform morphological closing to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    closed = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)

    # Convert to grayscale
    # closed = cv2.cvtColor(closed, cv2.COLOR_BGR2GRAY)

    # Find connected components in the binary image
    _, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    # Filter out small objects based on their area
    filtered = np.zeros_like(closed)
    for label in range(1, len(stats)):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_size:
            filtered[labels == label] = 255

    # Apply the filtered mask to the original image
    result = cv2.bitwise_and(closed, closed, mask=filtered)

    # Apply opening and closing
    kernel_opening = np.ones((10, 10), np.uint8)
    kernel_closing = np.ones((50, 50), np.uint8)
    
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel_opening)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_closing)
    return result