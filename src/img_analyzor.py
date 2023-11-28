

import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pyvisgraph as vg


def detecter_et_classifier_formes(image):
    # Charger l'image en niveaux de gris
    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Appliquer un aggrandisseur de formes 



    # Appliquer un flou pour réduire le bruit et détecter les contours
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.Canny(img, 50, 150)

    # Trouver les contours dans l'image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialiser une liste pour stocker les sommets et les classifications
    sommets_et_classes = []

    # Parcourir tous les contours
    for contour in contours:
        # Approximer le contour par une forme polygonale
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Déterminer la forme géométrique en fonction du nombre de sommets
        nb_sommets = len(approx)
        forme_geometrique = "autre forme"

        # Ajouter les sommets et la classification à la liste
        sommets_et_classes.append((approx, forme_geometrique))

    return sommets_et_classes
'''
def detecter_et_classifier_formes(image):
    # Appliquer un aggrandisseur de formes 
    # Convertir l'image dilatée en niveaux de gris
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer un flou pour réduire le bruit
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Utiliser la détection de contours avec l'algorithme de Canny
    img_canny = cv2.Canny(img_blur, 50, 150)

    # Trouver les contours dans l'image
    contours, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialiser une liste pour stocker les sommets et les classifications
    sommets_et_classes = []

    # Parcourir tous les contours
    for contour in contours:
        # Approximer le contour par une forme polygonale
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Déterminer la forme géométrique en fonction du nombre de sommets
        nb_sommets = len(approx)
        forme_geometrique = "autre forme"

        # Ajouter les sommets et la classification à la liste
        sommets_et_classes.append((approx, forme_geometrique))

    return sommets_et_classes

'''
def visualiser_chemin_plus_court(image, sommets_et_classes, chemin_plus_court):

    # Dessiner les sommets et les classifications sur l'image
    for sommets, classification in sommets_et_classes:
        cv2.drawContours(image, [sommets], 0, (0, 255, 0), 2)  # Dessiner les contours verts
        cv2.putText(image, classification, tuple(sommets[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Dessiner le chemin le plus court sur l'image
    for i in range(len(chemin_plus_court) - 1):
        start_pos = (int(chemin_plus_court[i].x), int(chemin_plus_court[i].y))
        end_pos = (int(chemin_plus_court[i + 1].x), int(chemin_plus_court[i + 1].y))
        cv2.line(image, start_pos, end_pos, (0, 0, 255), 2)  # Dessiner une ligne rouge

    # Afficher l'image avec les sommets, les classifications et le chemin le plus court
    cv2.imshow('Image avec sommets, classifications et chemin le plus court', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualiser_sommets(image, sommets_et_classes):

    # Dessiner les sommets et les classifications sur l'image
    for sommets, classification in sommets_et_classes:
        cv2.drawContours(image, [sommets], 0, (0, 255, 0), 2)  # Dessiner les contours verts
        cv2.putText(image, classification, tuple(sommets[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Afficher l'image avec les sommets et les classifications
    cv2.imshow('Image avec sommets et classifications', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def creer_polys(sommets_et_classes):
    polys = []

    for sommets, classification in sommets_et_classes:
        # Convertir les sommets de format OpenCV en format vg.Point
        sommets_vg = [vg.Point(x[0][0], x[0][1]) for x in sommets]

        # Ajouter les sommets à la liste des polys
        polys.append(sommets_vg)

    return polys

def convertir_chemin_en_array(chemin_plus_court):
    pathpoints = np.array([[point.x, point.y] for point in chemin_plus_court])
    return pathpoints




# Exemple d'utilisation avec une image
image_path = 'visi_img.png'
image = cv2.imread(image_path)

kernel = np.ones((5, 5), np.uint8)
image = cv2.erode(image, kernel, iterations=3)


sommets_et_classes = detecter_et_classifier_formes(image)

print(sommets_et_classes)



# Créer les polys à partir des sommets détectés
polys = creer_polys(sommets_et_classes)
graph = vg.VisGraph()
graph.build(polys)
shortest = graph.shortest_path(vg.Point(66, 107), vg.Point(217, 523))
print(shortest)
path = convertir_chemin_en_array(shortest)
print(path)



visualiser_chemin_plus_court(image, sommets_et_classes, shortest)

