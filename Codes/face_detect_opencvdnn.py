import cv2
import numpy as np
import os

# -----------------------------------------------
# Detection de visages par OpenCV DNN
# -----------------------------------------------

def euclidean_distance(a, b):
    # Assurez-vous que les vecteurs ont la même dimension
    assert len(a) == len(b)
    # Calcul de la somme des carrés des différences
    squared_diff = np.sum((a - b)**2)
    # Racine carrée de la somme des carrés
    distance = np.sqrt(squared_diff)
    return distance

# Fonction de detection
def detect_faces_OpenCV_DNN(net, frame):

    in_width = 96     # Resized image width passed to network
    in_height = 96    # Resized image height passed to network
    scale = 1/255     # Value scaling factor applied to input pixels
    mean = [0,0,0] # Mean BGR value subtracted from input image
    rgb = False          # True if model expects RGB inputs, otherwise it expects BGR

    h = frame.shape[0]
    w = frame.shape[1]

    # Mise au format blob pour le CNN
    blob = cv2.dnn.blobFromImage(frame, scale, (in_width, in_height), mean, rgb, crop=False)

    # Entrée du réseau
    net.setInput(blob)
    # calcul de la sortie
    detections = net.forward()

    return detections

# Lecture du réseau
fichier = "Codes\openface.nn4.small2.v1.t7"
net = cv2.dnn.readNetFromTorch(fichier)

# Chemin vers le dossier contenant les images de visages
folder_path = "Codes\visages"


# Liste pour stocker les vecteurs de caractéristiques pour chaque image
feature_vectors = []
file_names = []

# Parcourir toutes les images dans le dossier
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        # Chemin complet de l'image
        image_path = os.path.join(folder_path, filename)
        # Charger l'image avec OpenCV
        image = cv2.imread(image_path)
        # Extraction des caractéristiques faciales de l'image
        features = detect_faces_OpenCV_DNN(net, image)
        # Ajout du vecteur de caractéristiques à la liste
        feature_vectors.append(features)
        file_names.append(filename)

# Convertir la liste de vecteurs en tableau NumPy
feature_vectors = np.array(feature_vectors)

# boucle de lecture/traitement
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    # Extraction des caractéristiques faciales de la frame
    frame_features = detect_faces_OpenCV_DNN(net, frame)

    # Comparaison avec chaque vecteur dans la liste des vecteurs des images
    min_distance = float('inf')
    closest_match_index = None
    for i, features_database in enumerate(feature_vectors):
        # Calcul de la distance euclidienne entre les vecteurs
        distance = euclidean_distance(frame_features, features_database)
        # Mise à jour de la distance minimale et de l'index de la correspondance la plus proche
        if distance < min_distance:
            min_distance = distance
            closest_match_index = i

     # Affichage du résultat de la comparaison
    if min_distance < 0.99:  # Définir SEUIL_RECONNAISSANCE selon vos besoins
        closest_file_name = file_names[closest_match_index]
    else:
        closest_file_name = "Inconnu"

    
    print("Distance minimale:", min_distance, "Nom du fichier:", closest_file_name)

    # Affichage de la frame de la caméra
    cv2.imshow('Detection de visages par openCV DNN', frame)

    # Sortie de la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
video_capture.release()
cv2.destroyAllWindows()
