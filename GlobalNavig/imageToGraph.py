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

def find_contours(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.Canny(img, 50, 150)
    # Trouver les contours dans l'image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
    return contours

def find_polygons(img, threshold):
    contours = find_contours(img)
    # var inits
    polygons = {}
    count = 0
    
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
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
    return polygons

def distance(a,b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def find_connections(img_enlarged, nodelist, maxx, maxy): # polygons are enlarged polygons
    poly_connections = []
    connections = []
    # Add Polygons
    _, binary_image = cv2.threshold(img_enlarged, 127, 255, cv2.THRESH_BINARY)

    # Define the kernel size for erosion
    kernel_size = 3  # You can adjust this value
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply erosion to make structures slightly smaller
    eroded_image = cv2.erode(binary_image, kernel, iterations=6)   

    for i in nodelist:
        for j in nodelist:
            if j <= i:
                continue
            if check_line_through_white_area(eroded_image, nodelist[i], nodelist[j]):
                continue
            poly_connections.append([i, j, distance(nodelist[i], nodelist[j])])
    
    for i in poly_connections: # checks if any polygon lines go around the wall, which are not a valid path
        if (nodelist[i[0]][0]==0 and nodelist[i[1]][0]==0) or (nodelist[i[0]][0]==maxx and nodelist[i[1]][0]==maxx) or (nodelist[i[0]][1]==0 and nodelist[i[1]][1]==0) or (nodelist[i[0]][1]==maxy and nodelist[i[1]][1]==maxy):
            continue
        connections.append(i)
    return connections

def check_line_through_white_area(image, start_point, end_point):
    # Assuming the image is a binary image from erosion
    white_mask = (image == 255).astype(np.uint8)
 
    # Create an empty mask for the line, ensuring it's uint8 type
    line_mask = np.zeros_like(white_mask, dtype=np.uint8)

    # Draw the line on the mask
    cv2.line(line_mask, start_point, end_point, 1, thickness=1)

    # Check if any line pixel is also white in the image
    intersection = cv2.bitwise_and(white_mask, line_mask)
    return np.any(intersection)

def draw_enlarged(polygons, image_size):
    # Create a black background
    height, width = image_size[0], image_size[1]
    enlarged_img = np.zeros((height, width), dtype=np.uint8)

    # Draw white contours on the black background
    for _, contour_points in polygons.items():
        # Convert the points to NumPy array
        contour_array = np.array([point for _, point in contour_points.items()])

        # Fill the polygon with white color
        cv2.fillPoly(enlarged_img, [contour_array.astype(int)], color=(255))

    return enlarged_img

def find_enlarged_polygons(img, threshold):
    contours = find_contours(img)
    # Initialiser une liste pour stocker les sommets et les classifications
    polygons = {}
    poly_count =0
    node_count = 2
    # Parcourir tous les contours
    for contour in contours:
        # Approximer le contour par une forme polygonale
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
  
        polygons[poly_count]={}
        for i in approx:
            for j in i:
                skip = False
                if node_count>2: # skip nodes to close to each other
                    for k in polygons[poly_count].keys():
                        if length(vector(j.tolist(), polygons[poly_count][k]))<threshold:
                            skip=True
                            break
                if not skip:
                    polygons[poly_count][node_count]=j.tolist()
                    node_count += 1           
        poly_count += 1
    return polygons
