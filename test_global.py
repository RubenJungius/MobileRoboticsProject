from GlobalNavig.globalNavigation import *

start_node = 0
target_node = 1
offset = 105
start_pos = [1100,460]
target_pos = [120,300]
img = cv2.imread("Obstacle_map_filtered_2.png")
path, connections, nodelist = run_global(img, start_node, start_pos, target_node, target_pos, offset, 20) #, point_threshold, area_threshold)
draw_graph(img, connections, nodelist, path)
        