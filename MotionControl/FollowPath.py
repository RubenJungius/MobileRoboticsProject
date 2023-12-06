import numpy as np
import math
from timeit import default_timer as timer
import cv2 as cv
from tdmclient import ClientAsync, aw


# a tuner : d_projection qui definit le regard du robot), KPv et KPteta


KDteta = 2
KPteta = 70
segment_idx = 0
old_phi = 0

def distance(point1, point2):
    
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def vector_compute(point1, point2):

    vecteur = (point2[0] - point1[0], point2[1] - point1[1])
    return vecteur


def go_to_carrot(_position, _carrot, _teta, _margin) : 

    motorL = 0
    motorR = 0
    
    motorL = 90                 # forward speed
    motorR = 90                 

    global old_phi

    d = distance(_position, _carrot)
    vector_carrot = vector_compute(_position, _carrot)
    if d > _margin :
        phi =  math.atan2(vector_carrot[1],vector_carrot[0]) - _teta
        phi = adjust_angle(phi)
        # Apply P controller 
        motorL = motorL + phi * KPteta + (phi-old_phi)*KDteta
        motorR = motorR - phi * KPteta - (phi-old_phi)*KDteta
        old_phi = phi
    return motorL, motorR


def adjust_angle(_angle):

    while _angle > math.pi:
        _angle -= 2*math.pi
    while _angle < -math.pi:
        _angle += 2*math.pi
    return _angle




def follow_path(position, teta, path, path_has_been_done) :

    # Initialization of the variables
    projection = np.array([0,0])
    carrot = np.array([0,0])
    motorL = 0
    motorR = 0
    global segment_idx  #index of the target segment. (the segment wich the robot is following)
    
    # Set parameters
    margin = 5 # margin around the line  
    d_projection = 30   # distance from the robot to the projection 
    has_finished = 0    # flag to know if the robot has reached the last point
    limit_distance = 40     # distance in pixels to know if the robot has reached the end of an segment
    

    # Ckech if the path planning just came to be done 
    if path_has_been_done == 1 :
        segment_idx = 0
        path_has_been_done = 0

    # Check if the position has reached the end segment point
    if distance(position, path[segment_idx +1]) < limit_distance : 
        segment_idx += 1
        print('let s change segemnt')
        if segment_idx == len(path)-1 :
            has_finished = 1
            print('End')

    # Check if the Thymio has reached the end 
    if has_finished == 1:
        motorL = 0
        motorR = 0
        return motorL, motorR, has_finished, carrot
    else : 
        # step(1)
        projection = position + np.array([d_projection*math.cos(teta), d_projection*math.sin(teta)])
        # step(2)
        A = vector_compute(path[segment_idx], projection)
        B = vector_compute(path[segment_idx], path[segment_idx+1])
        Bnormal = B / np.linalg.norm(B)
        #step(3)
        sp = abs(np.dot(A,Bnormal))
        maxsp = distance(path[segment_idx],path[segment_idx+1])
        if sp >maxsp : 
            sp = maxsp
        #step(4)
        carrot = path[segment_idx] + Bnormal * sp
        
        
        motorL, motorR = go_to_carrot(position, carrot, teta, margin)
        return motorL, motorR, has_finished, carrot


