import cv2
import numpy as np

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
    
    # Prevent offset moving point offscreen
    new_point[0] = 0 if new_point[0]<0 else new_point[0]
    new_point[0] = img_shape[1] if new_point[0]>img_shape[1] else new_point[0]
    new_point[1] = 0 if new_point[1]<0 else new_point[1]
    new_point[1] = img_shape[0] if new_point[1]>img_shape[0] else new_point[1]
    
    return new_point

def enlarge_polygons(polygons, offset, img_shape):
    new_polygons = {}
    for polygon in polygons:
        count = 0
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
                        if length(vector(j, polygons[count][k]))<threshold:
                            skip=True
                    if skip:
                        continue  
                polygons[count][subcount]=j
                subcount += 1
        count += 1
    return polygons, count

def distance(a,b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def find_connections(nodelist, connections): # polygons are enlarged polygons

    return 0