import numpy as np
import termcolor
from colorama import Fore, Back, Style
import matplotlib.pyplot as plt
import time
import math

class kalman:
    def __init__(self, map_length_mm , mean_init):
        # New: Initialize variables for real-time plotting
        self.mean_history = []  # List to store the evolution of self.mean over time
        self.y_history = []
        self.time_steps = []
        # Initialize Matplotlib plot
        #plt.ion()  # Turn on interactive mode
        #self.fig, self.ax = plt.subplots()

        self.L = 95 # distance between wheels, mm
        self.thymio_speed_to_mms =1/3.1 
        self.map_length_mm = map_length_mm # initialized in main.py
        self.mean_init = mean_init     # use data from camera, multiply by constant to get actual first guess in mm
        self.pixel_to_mm = self.map_length_mm/1020      # length of final image is 1020 pixels
        self.mm_to_pixel = 1/self.pixel_to_mm
        self.mean =  self.pixel_to_mm * self.mean_init 
        # kalman.init_other_parameters(self)

        self.covar = np.zeros([3,3])
        self.A = np.eye(3)
        self.C = np.eye(3)

        self.q_x = 0.04
        self.q_y = 0.04 
        self.q_theta = 0.04
        self.Q = np.diag([self.q_x , self.q_y , self.q_theta])

        self.r_x = 0.01
        self.r_y = 0.01
        self.r_theta = 0.01
        self.R = np.diag([self.r_x , self.r_y , self.r_theta])

    # def init_other_parameters (self):
    #     self.pixel_to_mm = self.map_length_mm/1020      # length of final image is 1020 pixels
    #     self.mean = self.mean_init * self.pixel_to_mm

    def kalman_update_pose (self, vision_data , left_speed , right_speed , delta_t):        # vision_data is in pixels
        print(Fore.YELLOW)
        # print("vision_data: ", vision_data)
        # if (vision_data[0] is None) or (vision_data[1] is None) : #or (vision_data[2] is None):
        if vision_data is None:
            vision_bool = 0             # whether or not to consider the camera data
            vision_position = np.zeros([2,1])
            vision_theta = 0
            print("0 : NO vision")
        else:
            vision_bool = 1
            vision_position = self.pixel_to_mm * np.array([vision_data[0],vision_data[1]])          # vision_position  is in mm
            vision_theta = vision_data[2]
            print("1 : YES vision")
        print ("dt = ", delta_t)
        print("OLD KALMAN MEAN (mm) = ", self.mean)
        # print("old kalman covar= ",self.covar)
        lspeed = left_speed * self.thymio_speed_to_mms      # mms
        rspeed = right_speed * self.thymio_speed_to_mms      # mms
        v = (rspeed + lspeed)/2                 # mms
        w = - (rspeed - lspeed)/self.L            # rad/s ??
        print("right and left speed (OG units) = ", right_speed, ", ", left_speed)
        print("v (mms), w (unit?) = ", v , ", " , w)
        theta = self.mean[2]
        # B = delta_t * np.array( [[np.cos(theta) , 0] , [np.sin(theta) , 0 ] , [0 , 1]] ) # u vector is [v,w]
        u = np.array ([v,w])
        B = delta_t * np.array( [[np.cos(theta) , 0] , [np.sin(theta) , 0 ] , [0 , 1]] ) # u vector is [v,w]
        print(" u = ", u)
        print( "B = ", B)
        print( " B @ u = ", B@u )
        print("A = ", self.A)
        print("mean = ", self.mean)

        mean_predicted = self.A @ self.mean + B @ u
        mean_predicted
        x_predict = mean_predicted[0]
        y_predict = mean_predicted[1]
        theta_predict = mean_predicted[2]
        print("mean predicted (mm) = A@mean + B@u = ", mean_predicted)
        covar_predicted = self.A @ self.covar @ np.transpose(self.A)+ self.Q
        #fusion with visionprint("vision position (mm) = ", vision_position
        # y = np.transpose ( np.array([float(vision_position[0]),float(vision_position[1]),vision_theta]) )
        y = ( np.array([float(vision_position[0]),float(vision_position[1]),vision_theta]) )
        print("vision position (mm) = ", vision_position)
        innov =  y - self.C @ mean_predicted
        print("innov = ", innov)
        St = self.C @ covar_predicted @ np.transpose(self.C) + self.R
        Kt = vision_bool * covar_predicted @ np.transpose(self.C) @ np.linalg.inv(St)
        print("Kt = ",Kt)
        print("correction: kt@innov = ", Kt@innov)
        # updating estimations
        self.mean = mean_predicted + Kt @ innov
        self.mean [2] = (self.mean[2] + math.pi) % (2 * math.pi) - math.pi
        # w1 = 0.1
        # w2 = 0.9
        # self.mean = w1 * mean_predicted + w2 * y
        print("NEW KALMAN MEAN = ", self.mean)
        self.covar = ( np.eye(3) - Kt @ self.C ) @ covar_predicted
        # returning_mean_pix = self.mean
        # returning_mean_pix[0] = returning_mean_pix[0] * self.mm_to_pixel        # return in pixel values
        # returning_mean_pix[1] = returning_mean_pix[1] * self.mm_to_pixel
        # self.mean[0] = self.mean[0] * self.mm_to_pixel          # to pixel unit
        # self.mean[1] = self.mean[1] * self.mm_to_pixel
        print(Fore.RESET)

        # return returning_mean_pix, self.covar , x_predict , y_predict , theta_predict
        return x_predict , y_predict , theta_predict

        
        
