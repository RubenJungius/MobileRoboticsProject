import numpy as np
import math
from timeit import default_timer as timer
import cv2 as cv
from tdmclient import ClientAsync, aw
import matplotlib.pyplot as plt

from FollowPath import*
from Obstacle_avoid import*



client = ClientAsync()
node = aw(client.wait_for_node())
aw(node.lock())
#aw(client.sleep(4))
print("\nThymio connected hcacka")

# recieved data from others parts 
path_has_been_done = 0
pathpoints = np.array([[20,0],[10,10], [20,20]])
teta = (90) * math.pi/180
position = np.array([10,9.6])

# les choses indispensables pourque le programme marche sont : 
# les 4 variables ci- dessus
# une initialisation de la node du thymio 



### MOTION PART ###

# Bloc to compute the motors speed from obstacles
aw(node.wait_for_variables({"prox.horizontal"}))
prox_meas = node.v.prox.horizontal
motorL_obstacle, motorR_obstacle = LocalAvoidance(prox_meas)

# Bloc to compute the motors speed from path foloowing 
motorL_path, motorR_path = follow_path(position, teta, pathpoints, path_has_been_done)

# Compute the final motorL value and finale motorR value 
motorL = motorL_obstacle + motorL_path
motorR = motorR_obstacle + motorR_path

print(prox_meas[0], prox_meas[1], prox_meas[2], prox_meas[3])
print(motorL_obstacle,motorR_obstacle )
print(motorL_path, motorR_path)


# Limit the motor speed to 500 or -500
motorL = min(max(motorL, -500), 500)
motorR = min(max(motorR, -500), 500)


aw(node.set_variables(Msg_motors(motorL, motorR)))

print(motorL, motorR)

aw(client.sleep(4))
aw(node.set_variables(Msg_motors(0, 0)))
aw(node.unlock())


###     ###

# le truc est à tester sur le robot 

