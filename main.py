import cv2
import cv2.aruco as aruco
# from skimage.filters import threshold_otsu
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
import copy
from timeit import default_timer as timer
from tdmclient import ClientAsync, aw
from MotionControl.FollowPath import*
from LocalNavig.Obstacle_avoid import*

from Vision.vison_functions import*
  
from GlobalNavig.globalNavigation import*

from Filter.kalman import kalman

## A0 paper ratio --> a mesurer pour avoir les distances entre les points tu coco  
res_h = 1020
res_w = 720

map_length = 780     # mm
pixel_to_mm = map_length/1020
mm_to_pixel = 1/pixel_to_mm

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
    thymio_init = find_thymio(detected)
    (thymio_corners, initial_pos, initial_teta) = thymio_init
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

# Thymio connexion
client = ClientAsync()
node = aw(client.wait_for_node())
aw(node.lock())
aw(client.sleep(2))   
print("\nThymio connected success")

# Path planning var init
start_node = 0
target_node = 1
offset = 123        # unlarg obstacles
target = end_point  # define the target

# Motion Control var init
motorL=0
motorR=0
aw(node.set_variables(Msg_motors(motorL, motorR)))

#flags
path_has_been_done = 0
do_path = 1             # create a path (initially)

# run the detection of thymio pose to initialize the initial guess for the mean and orientation of the kalman filter
# ret, frame = cam.read()
# dst = crop_labyrinth(frame, M)
# dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
# detected = cv2.aruco.detectMarkers(dst_gray, arucoDict, parameters=arucoParams)
# thymio_pos = find_thymio(detected)     
thymio_pos = thymio_init
print("initial thymio pos : ", thymio_pos)
kalman_class = kalman(map_length_mm=map_length,mean_init=np.array([thymio_pos[1][0],thymio_pos[1][1], thymio_pos[2]])   )  #create an instance object for the class kalman()
# kalman_class.mean_init=[thymio_pos[1], thymio_pos[2]]    #mean is a vector for x,y,theta



curr_time=time.time()       # to not have the first dt very high

kalman_evol = []
vision_evol = []
time_evol = []
x_predicted_evol = []
y_predicted_evol = []
theta_predict_evol = []


start_time = time.time()

while True:

    

    ###### PATH PLANNING ######
    # Perform the path finding algorythm if the flag do_path is 1
    if do_path == 1 :
        path, connections, nodelist = run_global( obstacle_map, start_node, initial_pos.tolist(), target_node, target.tolist(), offset, 20, 40) #, point_threshold, area_threshold)
        positions_triees = {indice: nodelist[indice] for indice in path}        # nodelist gives all the nodes (directory of nodes)
        pathpoints = np.array(list(positions_triees.values()))[::-1]            # get a list of points coordinates from directorie 'node' and 'path' indexs
        cv2.destroyAllWindows()
        # Show map with all the posible paths and the chosen one 
        draw_graph(obstacle_map, connections, nodelist, path)
        print(pathpoints)

        do_path = 0
        path_has_been_done = 1





    ###### VISION #######
    
    ret, frame = cam.read()
    roi = get_ROI(frame, M)
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) #gray transform for aruco detection
    detected = cv2.aruco.detectMarkers(roi_gray, arucoDict, parameters=arucoParams)
    thymio_pos = find_thymio(detected)
    # get position of the robot and his orientation teta 
    c, position, teta = thymio_pos        
   


    ###### KALMAN FILTER ######
    print("--------------------------------------")
    # print("thymio_pos_camera: ", thymio_pos)
    if (position is not None) and (teta is not None):
        camera_pose = np.array([position[0],position[1],teta])
    else:
        camera_pose = None

    print("passed_thymio_pose_camera: ", camera_pose)
    aw(node.wait_for_variables({"motor.left.speed","motor.right.speed"}))
    l_speed = node.v.motor.left.speed
    r_speed = node.v.motor.right.speed
    prev_time = curr_time
    curr_time = time.time()
    delta_time = curr_time - prev_time
    print("kalman_class.mean before call =", kalman_class.mean)
    # kalman_pose , kalman_covariance , x_predicted , y_predicted , theta_predicted = kalman_class.kalman_update_pose (camera_pose,motorL,motorR, delta_time)        # "pose" contains position and orientation
    x_predicted , y_predicted , theta_predicted = kalman_class.kalman_update_pose (camera_pose,motorL,motorR, delta_time)        # "pose" contains position and orientation
    kalman_pose = copy.copy(kalman_class.mean)
    kalman_pose [0]= kalman_pose [0]* mm_to_pixel
    kalman_pose [1]= kalman_pose [1]* mm_to_pixel
    kalman_covariance = copy.copy(kalman_class.covar)
    x_predicted_evol.append(x_predicted)
    y_predicted_evol.append(y_predicted)
    theta_predict_evol.append(theta_predicted)
    print("kalman pose returned (pix): ", kalman_pose)
    print("kalman_class.mean after call =", kalman_class.mean)
    #plot
    kalman_evol.append(kalman_pose)
    vision_evol.append(camera_pose)
    current_timestep = time.time()
    time_evol.append(current_timestep)

    # thymio_pos = np.array([c , kalman_pose[0], kalman_pose[1], kalman_pose[2]])     # remerge into the thymio_pos variable to work with the rest of the code
    thymio_pos = (thymio_pos[0], np.array([kalman_pose[0],kalman_pose[1]]), kalman_pose[2])     
    # print("final thymio pos: ", thymio_pos)
    # print("thymio_pos2: ", thymio_pos)


        

    ###### MOTION CONTROL ######

    # Bloc to compute the motors speed from obstacles
    aw(node.wait_for_variables({"prox.horizontal"}))
    prox_meas = node.v.prox.horizontal
    motorL_obstacle, motorR_obstacle = LocalAvoidance(prox_meas)

    # Bloc to compute the motors speed from path foloowing 
    motorL_path, motorR_path, has_finished, carrot = follow_path(thymio_pos[1], thymio_pos[2], pathpoints, path_has_been_done)
    path_has_been_done = 0
    # Compute the final motorL value and finale motorR value 
    motorL = motorL_obstacle + motorL_path                      
    motorR = motorR_obstacle + motorR_path                      

    # Limit the motor speed to 500 or -500
    motorL = min(max(motorL, -500), 500)
    motorR = min(max(motorR, -500), 500)

    # Send speeds to Thymio. 1.01 and 0.98 are coefficients because of natural deiviation of the Thymio
    aw(node.set_variables(Msg_motors(motorL *1.01, motorR*0.98)))
    if has_finished == 1:
        aw(node.set_variables(Msg_motors(0, 0)))
        aw(node.unlock())
        break


    ##### SHOW MAP #####
    # Draw path 
    for point in pathpoints:
        cv2.circle(roi, tuple(np.array(point).astype(int)), 5, (0, 255, 0), -1)
    # Draw carrot (short term target of the thymio)
    cv2.circle(roi, tuple(np.array(carrot).astype(int)), 5, (255, 0, 0), -1)
    # Draw Thymio
    draw_thymio(roi, thymio_pos)                                                                           
    # Display the frame with detected markers and Thymio
    cv2.imshow('Frame with Detected Markers and Thymio', roi)




    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        aw(node.set_variables(Msg_motors(0, 0)))    # release the Thymio
        aw(node.unlock())
        break


cam.release()
cv2.destroyAllWindows()


# kalman_mean_history , kalman_y_history, kalman_timesteps = kalman_class.kalman_plot()
# plt.plot(kalman_timesteps , kalman_mean_history)

#plot:
# print("kalman_evol", kalman_evol)
# print("y_evol: ", vision_evol)
# print("timeevol: ", time_evol )
# print("kalman_evol[5]= ", kalman_evol[5])
# print("y_evol[5]=", vision_evol[5])

kalman_x_evol=[]
kalman_y_evol=[]
kalman_theta_evol=[]

vision_x_evol=[]
vision_y_evol=[]
vision_theta_evol=[]

for i in range (len(time_evol)) :
    kalman_i = kalman_evol[i] * pixel_to_mm
    kalman_x_evol.append(kalman_i[0])
    kalman_y_evol.append(kalman_i[1])
    kalman_theta_evol.append(kalman_i[2]/pixel_to_mm)

    # if all(vision_evol[i])== None:
    if vision_evol[i] is None: # and any(vision_evol[i]):
        vision_i = None
        vision_x_evol.append(None)
        vision_y_evol.append(None)
        vision_theta_evol.append(None)
    else:
        vision_i = vision_evol[i] * pixel_to_mm
    # if vision_i[0] == None:   
    #     vision_x_evol.append()
        vision_x_evol.append(vision_i[0])
        vision_y_evol.append(vision_i[1])
        vision_theta_evol.append(vision_i[2]/pixel_to_mm)


#plt.close('all')

# Clear all existing plots and graphics
#plt.clf()
#plt.close()

print('length time_evol= ', len(time_evol))
print('length kalman_evol= ', len(kalman_evol))



#plt.figure()
print("t1")

fig, axs = plt.subplots(2, 2, figsize=(15, 15))

# Plot the signals on each subplot
axs[0,0].plot(time_evol, kalman_x_evol , label='kalman x', color='blue')
axs[0,0].plot(time_evol, vision_x_evol , label='camera x', color='green')
axs[0,0].plot(time_evol, x_predicted_evol , label='predicted x', color='red')
axs[0,0].set_title('x evolution (mm)')
axs[0,0].legend()

axs[1,0].plot(time_evol, kalman_y_evol , label='kalman y', color='blue')
axs[1,0].plot(time_evol, vision_y_evol , label='camera y', color='green')
axs[1,0].plot(time_evol, y_predicted_evol , label='predicted y', color='red')
axs[1,0].set_title('y evolution (mm)')
axs[1,0].legend()

axs[0,1].plot(time_evol, kalman_theta_evol , label='kalman theta', color='blue')
axs[0,1].plot(time_evol, vision_theta_evol , label='camera theta', color='green')
axs[0,1].plot(time_evol, theta_predict_evol , label='predicted theta', color='red')
axs[0,1].set_title('theta evolution (rad)')
axs[0,1].legend()

axs[1,1].plot(kalman_x_evol, kalman_y_evol , label='kalman trajectory', color='blue')
axs[1,1].plot(vision_x_evol, vision_y_evol , label='camera trajectory', color='green')
axs[1,1].set_title('trajectory (mm)')
axs[1,1].legend()

plt.savefig('kalman_camera_evolution.png', bbox_inches='tight')

# Adjust layout to prevent subplot overlap
# plt.tight_layout()

# Display the plot
plt.show()
# print("tff")

