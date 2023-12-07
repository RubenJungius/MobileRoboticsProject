from GlobalNavig.globalNavigation import *

start_node = 0
target_node = 1
offset = 130
start_pos = [0,0]
target_pos = [600,600]
img = cv2.imread("Starting_map.png")
path, connections, nodelist = run_global(img, start_node, start_pos, target_node, target_pos, offset, 20, 40) #, point_threshold, area_threshold)
draw_graph(img, connections, nodelist, path)
        