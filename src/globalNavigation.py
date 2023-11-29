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

    '''
    # Artificial Border to prevent robot of running along Objects on edge
    maxy, maxx, _ = image.shape
    polys.append([vg.Point(0,0),vg.Point(maxx,0), vg.Point(maxx, -1), vg.Point(0,-1)])
    polys.append([vg.Point(0,0),vg.Point(0,maxy), vg.Point(-1,0), vg.Point(-1, maxy)])
    polys.append([vg.Point(maxx,0),vg.Point(maxx,maxy), vg.Point(maxx+1, 0), vg.Point(maxx+1, maxy)])
    polys.append([vg.Point(0,maxy),vg.Point(maxx,maxy), vg.Point(0,maxy+1), vg.Point(maxx, maxy+1)])
    '''
    
    graph = vg.VisGraph()
    graph.build(polys)
    
    return polygons, large_polygons, graph
    '''
    corners, nbCorners= imageToGraph.find_corners(img_grey, 0.5)
    corners[start_node]=start
    corners[target_node]=target
    nbCorners = nbCorners + 2
    
    connections = imageToGraph.find_connections(corners, nbCorners, img_grey)

    nodes = []
    init_graph = {}
    nodes.append(start_node)
    nodes.append(target_node)
    for x in corners:
        nodes.append(x)
        init_graph[x]={}
    for x in connections:
        init_graph[x[0]][x[1]]=x[2]

    graph = Graph.Graph(nodes, init_graph)
    return graph, nodes, nbCorners, corners, img_grey, connections
    '''
    
# TestCode
start_node = 0
target_node = 1
image_path = 'src\\sampleImg.jpeg'
image = cv2.imread(image_path)
offset = 30
start = [160,371]
target = [900,340]

polygons, large_polygons, graph= generateGraph(image, start, start_node, target, target_node, offset)
# graph, nodes, nbCorners, corners, img_grey, connections= generateGraph(image, start, start_node, target, target_node, offset)
# previous_nodes, shortest_path = Dijkstra.findPath(graph, start_node)

# path = Dijkstra.print_result(previous_nodes, shortest_path, start_node, target_node)
shortest_path = graph.shortest_path(vg.Point(start[0], start[1]), vg.Point(target[0], target[1]))
# dilated_image = dilate_img(image, offset)

print(large_polygons)

plt.figure()
# # plt.imshow(img_grey)
plt.imshow(image)
# # plt.imshow(dilated_image)

for i in polygons:
    for j in polygons[i]:
        plt.scatter(polygons[i][j][0], polygons[i][j][1], s=10, c='red', marker='o')
for i in large_polygons:
    for j in large_polygons[i]:
        plt.scatter(large_polygons[i][j][0], large_polygons[i][j][1], s=10, c='blue', marker='o')

count = 2
for i in large_polygons:
    for j in range(len(large_polygons[i])-1):
        plt.plot([large_polygons[i][count][0], large_polygons[i][count+1][0]], 
                 [large_polygons[i][count][1], large_polygons[i][count+1][1]], 'bo-')
        count += 1
    plt.plot([large_polygons[i][count-len(large_polygons[i])+1][0], large_polygons[i][count][0]], 
             [large_polygons[i][count-len(large_polygons[i])+1][1], large_polygons[i][count][1]], 'bo-')
    count += 1
for i in range(len(shortest_path)-1):
    plt.plot([shortest_path[i].x, shortest_path[i+1].x],[shortest_path[i].y, shortest_path[i+1].y], 'ro-')

plt.show()
