import numpy as np
import math
from timeit import default_timer as timer
import cv2 as cv
from tdmclient import ClientAsync, aw







#faire une fonction qui fait la local avoidance ! 

def LocalAvoidance(prox) : 


    NN = np.array([[2, 3, -4, -3, -2],[-2, -3, -4, 3, 2]])
    threshold = 500
    Gain = 0.01
    obstacle_detected = False

    for i in range(5):
        prox[i] = prox[i]
        if(prox[i] > threshold) :
            obstacle_detected = True

    if not(obstacle_detected) :
        return 0, 0

    elif obstacle_detected :
        Y = np.dot(NN, prox) * Gain
        motor_L = Y[0] 
        motor_R = Y[1]
        return motor_L, motor_R




def Msg_motors(_speedL, _speedR) : 

    msg = {
        "motor.left.target": [int(_speedL)],
        "motor.right.target": [int(_speedR)],
    }
    return msg
