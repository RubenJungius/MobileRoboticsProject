import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import pyvisgraph as vg
from timeit import default_timer as timer
from tdmclient import ClientAsync, aw

from img_analyzor import*
from FollowPath import*
from Obstacle_avoid import*



client = ClientAsync()
node = aw(client.wait_for_node())
aw(node.lock())
aw(client.sleep(4))   
print("\nThymio connected success")



path_has_been_done = 0
do_path = 1 

position =  np.array([66,107])
goal =  np.array([651,523])

###### Path planning #####

if do_path == 1 : 
    # on me donne obstacle 
    image_path = 'obstacle.png'
    image = cv2.imread(image_path)

    kernel = np.ones((5, 5), np.uint8)
    image = cv2.dilate(image, kernel, iterations=3)

    sommets_et_classes = detecter_et_classifier_formes(image)
    polys = creer_polys(sommets_et_classes)
    graph = vg.VisGraph()
    graph.build(polys)



    shortest = graph.shortest_path(vg.Point(position[0],position[1]), vg.Point(goal[0],goal[1]))

    pathpoints = convertir_chemin_en_array(shortest)
    print(pathpoints)
    do_path = 0
    path_has_been_done = 1


#### motion control ####

# get les data : 

#pathpoints = np.array([[20,0],[10,10], [20,20]])
# teta orientation du robot avec les abcisses
teta = (0) * math.pi/180



# Bloc to compute the motors speed from obstacles
aw(node.wait_for_variables({"prox.horizontal"}))
prox_meas = node.v.prox.horizontal

motorL_obstacle, motorR_obstacle = LocalAvoidance(prox_meas)

# Bloc to compute the motors speed from path foloowing 
motorL_path, motorR_path = follow_path(position, teta, pathpoints, path_has_been_done)

# Compute the final motorL value and finale motorR value 
motorL = motorL_obstacle + motorL_path
motorR = motorR_obstacle + motorR_path



# Limit the motor speed to 500 or -500
motorL = min(max(motorL, -500), 500)
motorR = min(max(motorR, -500), 500)


aw(node.set_variables(Msg_motors(motorL, motorR)))
print(motorL, motorR)


aw(client.sleep(4))
aw(node.set_variables(Msg_motors(0, 0)))
aw(node.unlock())
#ne pas oublier de dégager la node 