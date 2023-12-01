import cv2
import numpy as np
import matplotlib.pyplot as plt

vector = lambda a,b:[b[0]-a[0], b[1]-a[1]]
length = lambda a:np.sqrt(np.square(a[0])+np.square(a[1]))
scalarmult = lambda a,x:[a[0]*x, a[1]*x]
v_sum = lambda a,b:[a[0]+b[0], a[1]+b[1]]

def offset_point(center, a, b, offset, img_shape):
    v1 = vector(a, center)
    v1 = scalarmult(v1, 1/length(v1)*offset)
    v2 = vector(b, center)
    v2 = scalarmult(v2, 1/length(v2)*offset)
    new_point = v_sum(v_sum(center, v1), v2)
    
    new_point[0]=int(new_point[0])
    new_point[1]=int(new_point[1])

    # Prevent offset moving point offscreen
    new_point[0] = 0 if new_point[0]<0 else new_point[0]
    new_point[0] = img_shape[1] if new_point[0]>img_shape[1] else new_point[0]
    new_point[1] = 0 if new_point[1]<0 else new_point[1]
    new_point[1] = img_shape[0] if new_point[1]>img_shape[0] else new_point[1]
    
    return new_point

def enlarge_polygons(polygons, offset, img_shape):
    new_polygons = {}
    count = 2
    for polygon in polygons:
        new_polygons[polygon] = {}
        nbKeys = len(polygons[polygon].keys())
        
        new_polygons[polygon][count] = offset_point(polygons[polygon][0], polygons[polygon][1], 
                                                         polygons[polygon][nbKeys-1], offset, img_shape)
        count += 1
        for i in range(1, nbKeys-1):
            new_polygons[polygon][count] = offset_point(polygons[polygon][i], polygons[polygon][i+1], 
                                                             polygons[polygon][i-1], offset, img_shape)
            count += 1
        new_polygons[polygon][count] = offset_point(polygons[polygon][nbKeys-1], polygons[polygon][nbKeys-2], 
                                                         polygons[polygon][0], offset, img_shape)
        count += 1
    return new_polygons

def find_polygons(img, threshold):
    # Appliquer un flou pour réduire le bruit et détecter les contours
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.Canny(img, 50, 150)

    # Trouver les contours dans l'image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialiser une liste pour stocker les sommets et les classifications
    polygons = {}
    count = 0
    
    # Parcourir tous les contours
    for contour in contours:
        # Approximer le contour par une forme polygonale
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        hull = cv2.convexHull(approx)
        
        polygons[count]={}
        subcount=0
        for i in hull:
            for j in i:
                if subcount > 0:
                    skip = False
                    for k in range(subcount):
                        break
                        if length(vector(j, polygons[count][k]))<threshold:
                            skip=True
                    if skip:
                        continue  
                polygons[count][subcount]=j
                subcount += 1
        count += 1
    return polygons

def distance(a,b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def find_connections(nodelist, polygons, maxx, maxy, image): # polygons are enlarged polygons
    poly_connections = []
    connections = []
    for i in polygons:
        for j in polygons[i]:
            try:
                poly_connections.append([j,j+1,distance(polygons[i][j], polygons[i][j+1])])
            except:
                poly_connections.append([j,j-len(polygons[i])+1,distance(polygons[i][j], polygons[i][j-len(polygons[i])+1])])
    
    for i in nodelist:
        for j in nodelist:
            if i == j: 
                continue
            samePoly = False
            for a in polygons:
                keyList = list(polygons[a].keys())
                if i in keyList and j in keyList:
                    samePoly = True
                    break
            if samePoly:
                continue
            
            isConnection = True
            for k in poly_connections:
                if intersect(nodelist[i], nodelist[j], nodelist[k[0]], nodelist[k[1]]):
                    '''
                    print(nodelist[i], nodelist[j], nodelist[k[0]], nodelist[k[1]])
                    plt.clf()
                    plt.imshow(image)
                    plt.plot([nodelist[i][0], nodelist[j][0]], [nodelist[i][1], nodelist[j][1]], color="b")
                    plt.plot([nodelist[k[0]][0], nodelist[k[1]][0]], [nodelist[k[0]][1], nodelist[k[1]][1]], color="r")
                    plt.show(block=False)
                    plt.pause(0.001)
                    input("Press Enter to continue...")
                    '''

                    isConnection = False
                    break
            if isConnection:
                connections.append([i,j, distance(nodelist[i], nodelist[j])])
    
    for i in poly_connections:
        if (nodelist[i[0]][0]==0 and nodelist[i[1]][0]==0) or (nodelist[i[0]][0]==maxx and nodelist[i[1]][0]==maxx) or (nodelist[i[0]][1]==0 and nodelist[i[1]][1]==0) or (nodelist[i[0]][1]==maxy and nodelist[i[1]][1]==maxy):
            continue
        connections.append(i)
    return connections

# Functions to find if 2 lines intersect
def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A,B,C,D):
    if A == C or A == D or B == C or B == D:
        return False
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

# Try2
def orientation(p, q, r):
    """
    Function to find the orientation of triplet (p, q, r).
    Returns:
     0 : Collinear points
     1 : Clockwise points
    -1 : Counterclockwise points
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0  # Collinear
    return 1 if val > 0 else -1  # Clockwise or Counterclockwise

def on_segment(p, q, r):
    """
    Function to check if point q lies on line segment 'pr'.
    """
    return q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and \
           q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])

def do_intersect(p1, q1, p2, q2):
    """
    Function to check if two line segments intersect.
    """

    if p1 == p2 or p1 == q2 or q1 == p2 or q1 == q2:
        return False

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special Cases

    # p1, q1, and p2 are collinear and p2 lies on the segment p1q1
    if o1 == 0 and on_segment(p1, p2, q1):
        return False

    # p1, q1, and q2 are collinear and q2 lies on the segment p1q1
    if o2 == 0 and on_segment(p1, q2, q1):
        return False

    # p2, q2, and p1 are collinear and p1 lies on the segment p2q2
    if o3 == 0 and on_segment(p2, p1, q2):
        return False

    # p2, q2, and q1 are collinear and q1 lies on the segment p2q2
    if o4 == 0 and on_segment(p2, q1, q2):
        return False

    return False  # Doesn't fall in any of the above cases