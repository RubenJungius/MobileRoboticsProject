import Graph
import Dijkstra
import imageToGraph
import pyvisgraph as vg
import cv2
import matplotlib.pyplot as plt
import numpy as np

def generateGraph(image, start, start_node, target, target_node, offset):
    
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    polygons, nbPolygons = imageToGraph.find_polygons(img_grey, threshold=20.0)
    large_polygons = imageToGraph.enlarge_polygons(polygons, offset, image.shape)
    
    polys = []
    for i in large_polygons:
        corners = []
        for j in large_polygons[i]:
            corners.append(vg.Point(large_polygons[i][j][0], large_polygons[i][j][1]))
        polys.append(corners)

    nodes_list = {}
    nodes_list[start_node]=start
    nodes_list[target_node]=target
    for i in large_polygons:
        for j in large_polygons[i]:
            nodes_list[j]=large_polygons[i][j]
    # print(nodes_list)
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
    # Build vg Graph
    '''
    graph = vg.VisGraph()
    graph.build(polys)
    '''
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

# shortest_path = graph.shortest_path(vg.Point(start[0], start[1]), vg.Point(target[0], target[1]))
previous_node, shortest_path = Dijkstra.findPath(graph, start_node)
path = Dijkstra.get_shortest(previous_node, shortest_path, start_node, target_node)

plt.figure()
plt.imshow(image)

for i in polygons:
    for j in polygons[i]:
        plt.scatter(polygons[i][j][0], polygons[i][j][1], s=10, c='red', marker='o')
for i in large_polygons:
    for j in large_polygons[i]:
        plt.scatter(large_polygons[i][j][0], large_polygons[i][j][1], s=10, c='blue', marker='o')

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

# # Shortest path with vg
'''
for i in range(len(shortest_path)-1):
    plt.plot([shortest_path[i].x, shortest_path[i+1].x],[shortest_path[i].y, shortest_path[i+1].y], 'ro-')
'''
# Plot shortest Path
for i in range(len(path)-1):
    plt.plot([nodelist[path[i]][0], nodelist[path[i+1]][0]],[nodelist[path[i]][1], nodelist[path[i+1]][1]], 'bo-')
plt.show()
