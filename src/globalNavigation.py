from src.Graph import*
from src.Dijkstra import*
from src.imageToGraph import*

import cv2
import matplotlib.pyplot as plt
import numpy as np

def generateGraph(image, start, start_node, target, target_node, offset, threshold, area_threshold):
    
    #img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    polygons = find_polygons(image, threshold, area_threshold)
    large_polygons = enlarge_polygons(polygons, offset, image.shape)
    
    nodes_list = {}
    nodes_list[start_node]=start
    nodes_list[target_node]=target
    for i in large_polygons:
        for j in large_polygons[i]:
            nodes_list[j]=large_polygons[i][j]
    
    connections = find_connections(nodes_list, large_polygons, maxx=image.shape[1], maxy=image.shape[0])
    
    ## here build graph with graphclass
    nodes = []
    init_graph = {}
    for i in nodes_list:
        nodes.append(i)
        init_graph[i]={}
    for i in connections:
        init_graph[i[0]][i[1]]=i[2]
    graph= Graph(nodes, init_graph)

    return graph, connections, nodes_list
    
def draw_graph(image, connections, nodelist, path):
    plt.figure()
    plt.imshow(image)
    for i in connections:
        plt.plot([nodelist[i[0]][0], nodelist[i[1]][0]], [nodelist[i[0]][1], nodelist[i[1]][1]], 'ro-')

    # Plot shortest Path
    for i in range(len(path)-1):
        plt.plot([nodelist[path[i]][0], nodelist[path[i+1]][0]],[nodelist[path[i]][1], nodelist[path[i+1]][1]], 'bo-')
    plt.show()

def run_global(image, start_node, start, target_node, target, offset, threshold):
    graph, connections, nodelist= generateGraph(image, start, start_node, target, target_node, offset, threshold)
    previous_node, shortest_path = findPath(graph, start_node)
    path = get_shortest(previous_node, start_node, target_node)
    return path, connections, nodelist

def formatter_positions(dictionnary):
    # Obtenez une liste d'indices basée sur les clés du dictionnaire
    indices = list(dictionnary.keys())

    # Créez un tableau numpy à partir des valeurs du dictionnaire dans le format spécifié
    positions_formattees = np.array([dictionnary[indice] for indice in indices])

    return positions_formattees



'''

# TestCode
start_node = 0
target_node = 1
image_path = 'Obstacle_map.png'
image = cv2.imread(image_path)
offset = 60
start = [100,500]
target = [820,370]

path, connections, nodelist = run_global(image, start_node, start, target_node, target, offset)
draw_graph(image, connections, nodelist, path)
#path_points = formatter_positions(nodelist)


positions_triees = {indice: nodelist[indice] for indice in path}
path_points = np.array(list(positions_triees.values()))[::-1]


'''
