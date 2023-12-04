import cv2
import cv2.aruco as aruco
from skimage.filters import threshold_otsu
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


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


############################ Code ###############################

# Open Camera
cam = cv2.VideoCapture(1)
if not cam.isOpened():
    print("Error: Unable to open the camera.")
else:
    print("\nSearching corners and correct perspective")

cam.set(cv2.CAP_PROP_FRAME_WIDTH, res_h)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, res_w)

#search for the 4 corner and the right transformation M
M=corners_calibration(cam)

print("\nMap computing")
ret, frame = cam.read()
map = crop_labyrinth(frame, M)
cv2.imwrite("Global_map.png", map)

print("\nFind Thymio")
thymio_pos = None
initial_pos= None
initial_teta = None
#iterate over the few first frame to look for thymio (id=5)
while initial_pos is None and initial_teta is None:
    ret, frame = cam.read()
    dst = crop_labyrinth(frame, M)
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    detected = cv2.aruco.detectMarkers(dst_gray, arucoDict, parameters=arucoParams)
    (thymio_corners, initial_pos, initial_teta) = find_thymio(detected)

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

# Thymio connexion
client = ClientAsync()
node = aw(client.wait_for_node())
aw(node.lock())
aw(client.sleep(2))   
print("\nThymio connected success")

# var declarations
start_node = 0
target_node = 1
offset = 70
target =  np.array([900,400])
aw(node.set_variables(Msg_motors(0, 0)))
last_position = initial_pos
last_teta = initial_teta
point_threshold = 20
area_threshold = 40

#flags
path_has_been_done = 0
do_path = 1 


while True:
    ret, frame = cam.read()
    dst = crop_labyrinth(frame, M)
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    detected = cv2.aruco.detectMarkers(dst_gray, arucoDict, parameters=arucoParams)
    thymio_pos = find_thymio(detected)
    draw_thymio(dst, thymio_pos)
    # Display the frame with detected markers and Thymio
    cv2.imshow('Frame with Detected Markers and Thymio', dst)



    ###### PATH PLANNING ######
    # compute position 
    c, position, teta = thymio_pos
    if position is None : 
        position = last_position
    last_position = position 
    if teta is None : 
        teta = last_teta
    last_teta = teta


    if do_path == 1 :

        path, connections, nodelist = run_global(obstacle_map, start_node, position, target_node, target, offset, point_threshold, area_threshold)
        positions_triees = {indice: nodelist[indice] for indice in path}
        pathpoints = np.array(list(positions_triees.values()))[::-1]
        draw_graph(obstacle_map, connections, nodelist, path)
        print(pathpoints)

        do_path = 0
        path_has_been_done = 1
        

    ###### MATION CONTROL ######

    # Bloc to compute the motors speed from obstacles
    aw(node.wait_for_variables({"prox.horizontal"}))
    prox_meas = node.v.prox.horizontal
    motorL_obstacle, motorR_obstacle = LocalAvoidance(prox_meas)

    # Bloc to compute the motors speed from path foloowing 
    motorL_path, motorR_path, has_finished, carrot = follow_path(position, teta, pathpoints, path_has_been_done)
    path_has_been_done = 0
    # Compute the final motorL value and finale motorR value 
    motorL = motorL_obstacle + motorL_path
    motorR = motorR_obstacle + motorR_path

    # Limit the motor speed to 500 or -500
    motorL = min(max(motorL, -500), 500)
    motorR = min(max(motorR, -500), 500)

    aw(node.set_variables(Msg_motors(motorL, motorR)))
    if has_finished == 1:
        aw(node.set_variables(Msg_motors(0, 0)))
        aw(node.unlock())
        break







    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        aw(node.set_variables(Msg_motors(0, 0)))
        aw(node.unlock())
        break


# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()