import numpy as np
import math
from timeit import default_timer as timer
import cv2 as cv
from tdmclient import ClientAsync, aw


# a tuner : d_projection qui definit le regard du robot), KPv et KPteta

'''
has_finished = 0
KPv = 12 
KPteta = 10
path_has_been_done = 1 
margin = 1 # marge de protection autour de la ligne 
d_projection = 2


segment_idx = 0
limit_distance = 1
'''

KPv = 12 
KPteta = 10
segment_idx = 0
has_finished = 0

def distance(point1, point2):
    
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def vector_compute(point1, point2):

    vecteur = (point2[0] - point1[0], point2[1] - point1[1])
    return vecteur


def go_to_carrot(_position, _carrot, _teta, _Bnormal, _margin) : 

    d = distance(_position, _carrot)
    motorL = d * KPv
    motorR = d * KPv

    ## version pas de ralentissement ; 
    # d = distance(_position, _projection)
    #motorL = d * KPv
    #motorR = d * KPv

    if d > _margin :
        phi = _teta - math.atan2(_Bnormal[1],_Bnormal[0])
        phi = adjust_angle(phi)
        #print(_teta)
        #print('angle Bnormal : ')
        #print(math.atan2(_Bnormal[1],_Bnormal[0]))
        #print(phi)
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

    
    margin = 1 # marge de protection autour de la ligne 
    d_projection = 2

    global has_finished
    global segment_idx 
    limit_distance = 1



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

    

    if has_finished == 1:
        motorL = 0
        motorR = 0
        return motorL, motorR
    else : 
        # project mon point sur le segment que je suis 
        projection = position + np.array([d_projection*math.cos(teta), d_projection*math.sin(teta)])
        A = vector_compute(path[segment_idx], projection)
        B = vector_compute(path[segment_idx], path[segment_idx+1])
        Bnormal = B / np.linalg.norm(B)
        #print(B)
        #print('Bnormal =')
        #print(Bnormal)
        sp = np.dot(A,Bnormal)
        #print(sp)
        carrot = path[segment_idx] + Bnormal * abs(sp)
        #print(carrot)
        

        #print('angle = ' + str(phi)+ str(phi1))
        motorL, motorR = go_to_carrot(position, carrot, teta, Bnormal, margin)
        #print('motorL = ' + str(motorL)+ 'motorR = ' + str(motorR))
        return motorL, motorR


