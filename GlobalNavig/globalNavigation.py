from GlobalNavig.Graph import*
from GlobalNavig.Dijkstra import*
from GlobalNavig.imageToGraph import*

import cv2
import matplotlib.pyplot as plt
import numpy as np

def generateGraph(image, start, start_node, target, target_node, offset, threshold, area_threshold):
    
    #img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    polygons = find_polygons(image, threshold, area_threshold)
    large_polygons = enlarge_polygons(polygons, offset, image.shape)
    
###
    enlarged_img = draw_enlarged(large_polygons, image.shape)
    large_polygons = find_enlarged_polygons(enlarged_img, threshold, area_threshold)
    '''
    print(large_polygons)
    plt.figure()
    plt.imshow(enlarged_img)
    for i in large_polygons:
        for j in large_polygons[i]:
            plt.scatter(large_polygons[i][j][0], large_polygons[i][j][1])
    plt.show()
    '''
###
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

def run_global(image, start_node, start, target_node, target, offset, threshold, area_threshold):
    graph, connections, nodelist= generateGraph(image, start, start_node, target, target_node, offset, threshold, area_threshold)
    previous_node, shortest_path = findPath(graph, start_node)
    path = get_shortest(previous_node, start_node, target_node)
    return path, connections, nodelist

def formatter_positions(dictionnary):
    # Obtenez une liste d'indices basée sur les clés du dictionnaire
    indices = list(dictionnary.keys())

    # Créez un tableau numpy à partir des valeurs du dictionnaire dans le format spécifié
    positions_formattees = np.array([dictionnary[indice] for indice in indices])

    return positions_formattees




