import numpy as np
import termcolor
from colorama import Fore, Back, Style
import matplotlib.pyplot as plt
import time
import math

class kalman:

    def __init__(self, map_length_mm , mean_init):

        self.L = 95 # distance between wheels, mm
        self.thymio_speed_to_mms =1/3.1 
        self.map_length_mm = map_length_mm 
        self.mean_init = mean_init # use data from camera to initialize the states
        self.pixel_to_mm = self.map_length_mm/1020  # length of final image is 1020 pixels
        self.mm_to_pixel = 1/self.pixel_to_mm
        self.mean =  self.pixel_to_mm * self.mean_init  # get the initial states in mm

        self.covar = np.zeros([3,3])
        self.A = np.eye(3)
        self.C = np.eye(3)

        self.q_x = 0.04
        self.q_y = 0.04 
        self.q_theta = 0.08
        self.Q = np.diag([self.q_x , self.q_y , self.q_theta])

        self.r_x = 0.01721
        self.r_y = 0.00577
        self.r_theta = 3.966e-6
        self.R = np.diag([self.r_x , self.r_y , self.r_theta])

    def kalman_update_pose (self, vision_data , left_speed , right_speed , delta_t): # vision_data is in pixels, speeds in thymio unit

        if vision_data is None:  # if camera is hidden
            vision_bool = 0 # whether or not to consider the camera data (0 means don't)
            vision_position = np.zeros([2,1]) # set to anything except None to avoid errors
            vision_theta = 0
        else:
            vision_bool = 1
            vision_position = self.pixel_to_mm * np.array([vision_data[0],vision_data[1]]) # vision_position becomes in mm
            vision_theta = vision_data[2]

        lspeed = left_speed * self.thymio_speed_to_mms # mm/s
        rspeed = right_speed * self.thymio_speed_to_mms # mm/s
        v = (rspeed + lspeed)/2 # mm/s           
        w = (lspeed - rspeed)/self.L  # rad/s
        theta = self.mean[2]
        u = np.array ([v,w])
        B = delta_t * np.array( [[np.cos(theta) , 0] , [np.sin(theta) , 0 ] , [0 , 1]] ) # B changes with theta, so recompute B every iteration        
        mean_predicted = self.A @ self.mean + B @ u  # prediction
        x_predict = mean_predicted[0] # for plotting at the end the evolution of the predicted data before fusing with vision
        y_predict = mean_predicted[1]
        theta_predict = mean_predicted[2]
        covar_predicted = self.A @ self.covar @ np.transpose(self.A)+ self.Q

        # vision
        y = ( np.array([float(vision_position[0]),float(vision_position[1]),vision_theta]) )   # change the format of vision_position
        innov =  y - self.C @ mean_predicted # innovation = measurements - prediction
        St = self.C @ covar_predicted @ np.transpose(self.C) + self.R
        Kt = vision_bool * covar_predicted @ np.transpose(self.C) @ np.linalg.inv(St) 

        # updating estimations and covariance
        self.mean = mean_predicted + Kt @ innov # if no vision Kt=0 so only use predicted data
        self.mean [2] = (self.mean[2] + math.pi) % (2 * math.pi) - math.pi  # keep theta between -pi and pi
        self.covar = ( np.eye(3) - Kt @ self.C ) @ covar_predicted

        return x_predict , y_predict , theta_predict        