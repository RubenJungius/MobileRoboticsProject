import Graph
import Dijkstra
import imageToGraph
import cv2
import matplotlib.pyplot as plt
import numpy as np

def generateGraph(image, start, start_node, target, target_node, offset):
    
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    polygons = imageToGraph.find_polygons(img_grey, threshold=20.0)
    large_polygons = imageToGraph.enlarge_polygons(polygons, offset, image.shape)
    
    nodes_list = {}
    nodes_list[start_node]=start
    nodes_list[target_node]=target
    for i in large_polygons:
        for j in large_polygons[i]:
            nodes_list[j]=large_polygons[i][j]
    
    connections = imageToGraph.find_connections(nodes_list, large_polygons, maxx=image.shape[1], maxy=image.shape[0])
    
    ## here build graph with graphclass
    nodes = []
    init_graph = {}
    for i in nodes_list:
        nodes.append(i)
        init_graph[i]={}
    for i in connections:
        init_graph[i[0]][i[1]]=i[2]
    graph= Graph.Graph(nodes, init_graph)

    return polygons, large_polygons, graph, connections, nodes_list
    
# TestCode
start_node = 0
target_node = 1
image_path = 'src\\sampleImg.jpeg'
image = cv2.imread(image_path)
offset = 30
start = [160,700]
target = [950,180]

polygons, large_polygons, graph, connections, nodelist= generateGraph(image, start, start_node, target, target_node, offset)
previous_node, shortest_path = Dijkstra.findPath(graph, start_node)
path = Dijkstra.get_shortest(previous_node, start_node, target_node)

plt.figure()
plt.imshow(image)

# Draw Enlarged Polygons
'''
count = 2
for i in large_polygons:
    for j in range(len(large_polygons[i])-1):
        plt.plot([large_polygons[i][count][0], large_polygons[i][count+1][0]], 
                 [large_polygons[i][count][1], large_polygons[i][count+1][1]], 'bo-')
        count += 1
    plt.plot([large_polygons[i][count-len(large_polygons[i])+1][0], large_polygons[i][count][0]], 
             [large_polygons[i][count-len(large_polygons[i])+1][1], large_polygons[i][count][1]], 'bo-')
    count += 1
'''
for i in connections:
    plt.plot([nodelist[i[0]][0], nodelist[i[1]][0]], [nodelist[i[0]][1], nodelist[i[1]][1]], 'ro-')

# Plot shortest Path
for i in range(len(path)-1):
    plt.plot([nodelist[path[i]][0], nodelist[path[i+1]][0]],[nodelist[path[i]][1], nodelist[path[i+1]][1]], 'bo-')
plt.show()
