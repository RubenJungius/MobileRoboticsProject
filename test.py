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



client = ClientAsync()
node = aw(client.wait_for_node())
aw(node.lock())
aw(client.sleep(2))   

while True : 
    
    aw(node.wait_for_variables({"button.forward"}))
    
    print(node.v.button.forward)
    aw(client.sleep(1)) 
        # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # release the Thymio
        aw(node.unlock())
        break