import Graph
import Dijkstra

def readFile():
    FILE = 'src\\sampleInput.txt'
    f = open(FILE,'r')

    for i in f:
        x = i.split()
        if x[0] not in nodes:
            nodes.append(x[0])
            init_graph[x[0]]={}
        if x[1] not in nodes: # needed for target node, ugly but idc rn
            nodes.append(x[1])
            init_graph[x[1]]={}
        init_graph[x[0]][x[1]] = int(x[2])

nodes = []
init_graph = {}
start_node = 'S'
target_node = 'G'

readFile() # to be replaced with a function reading the matrix
graph = Graph.Graph(nodes, init_graph)
previous_nodes, shortest_path = Dijkstra.findPath(graph, start_node)
Dijkstra.print_result(previous_nodes, shortest_path, start_node, target_node)