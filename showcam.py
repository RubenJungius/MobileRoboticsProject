import cv2
import numpy as np
import matplotlib.pyplot as plt
# from skimage.filters import threshold_otsu

# Initialiser la capture vidéo à partir de la caméra numéro 1
cam = cv2.VideoCapture(1)

# Vérifier si la caméra a été ouverte avec succès
if not cam.isOpened():
    print("Erreur: Impossible d'ouvrir la caméra.")
    exit()

# Boucle de capture vidéo
while True:
    # Lire le frame depuis la caméra
    ret, frame = cam.read()

    # Vérifier si la lecture du frame a réussi
    if not ret:
        print("Erreur lors de la lecture du frame.")
        break

    # Afficher le frame capturé
    cv2.imshow('Camera', frame)

    # Attendre 1 milliseconde et vérifier si une touche est enfoncée
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Appuyez sur la touche 'q' pour quitter la boucle
        break

# Libérer la ressource de la caméra et fermer la fenêtre
cam.release()
cv2.destroyAllWindows()
