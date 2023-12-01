import numpy as np
import math
from timeit import default_timer as timer
import cv2 as cv
from tdmclient import ClientAsync, aw


# a tuner : d_projection qui definit le regard du robot), KPv et KPteta


KPv = 2
KPteta = 70
segment_idx = 0

def distance(point1, point2):
    
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def vector_compute(point1, point2):

    vecteur = (point2[0] - point1[0], point2[1] - point1[1])
    return vecteur


def go_to_carrot(_position, _carrot, _teta, _Bnormal, _margin) : 

    motorL = 0
    motorR = 0
    d = distance(_position, _carrot)
    motorL = 70
    motorR = 70

    ## version pas de ralentissement ; 
    #d = distance(_position, _projection)
    #motorL = d * KPv
    #motorR = d * KPv
    vector_carrot = vector_compute(_position, _carrot)
    if d > _margin :
        phi =  math.atan2(vector_carrot[1],vector_carrot[0]) - _teta
        phi = adjust_angle(phi)
        #phi = - phi # Inversion thanks to the fact that computation are made in an other base that images
        motorL = motorL + phi * KPteta
        motorR = motorR - phi * KPteta

    return motorL, motorR


def adjust_angle(_angle):

    while _angle > math.pi:
        _angle -= 2*math.pi
    while _angle < -math.pi:
        _angle += 2*math.pi
    return _angle




def follow_path(position, teta, path, path_has_been_done) :


    projection = np.array([0,0])
    carrot = np.array([0,0])
    motorL = 0
    motorR = 0
    global segment_idx
    margin = 5 # marge de protection autour de la ligne 
    d_projection = 30

    has_finished = 0
    limit_distance = 40
    


    # Ckech if the path planning just came to be done 
    if path_has_been_done == 1 :
        segment_idx = 0
        path_has_been_done = 0

    # Check if the position has reached the end segment point
    if distance(position, path[segment_idx +1]) < limit_distance : 
        segment_idx += 1
        print('go changer de segemnt')
        if segment_idx == len(path)-1 :
            has_finished = 1
            print('cest finit')

    #print(position, path[segment_idx +1])

    if has_finished == 1:
        motorL = 0
        motorR = 0
        return motorL, motorR, has_finished, carrot
    else : 
        # project mon point sur le segment que je suis 
        projection = position + np.array([d_projection*math.cos(teta), d_projection*math.sin(teta)])
        A = vector_compute(path[segment_idx], projection)
        B = vector_compute(path[segment_idx], path[segment_idx+1])
        Bnormal = B / np.linalg.norm(B)
        #print(B)
        #print('Bnormal =')
        #print(Bnormal)
        sp = abs(np.dot(A,Bnormal))
        maxsp = distance(path[segment_idx],path[segment_idx+1])
        if sp >maxsp : 
            sp = maxsp
        #print(sp)
        carrot = path[segment_idx] + Bnormal * abs(sp)
        #print(carrot)
        

        #print('angle = ' + str(phi)+ str(phi1))
        motorL, motorR = go_to_carrot(position, carrot, teta, Bnormal, margin)
        #print('motorL = ' + str(motorL)+ 'motorR = ' + str(motorR))
        return motorL, motorR, has_finished, carrot


