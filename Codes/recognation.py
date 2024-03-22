import cv2
import numpy as np
import os

# -----------------------------------------------
# Detection de visages par OpenCV DNN avec YOLOv3
# -----------------------------------------------

def load_model(config_path, weight_path):
    return cv2.dnn.readNetFromDarknet(config_path, weight_path)

def detect_faces_YOLOv3(net, frame, min_conf=0.5, nms_threshold=0.4):
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > min_conf:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return frame,boxes

# Lecture du modèle YOLOv3
model_weights = "E:\mineur\TP6\Codes-20240228\Codes\yolov3-tiny_obj_best.weights"
model_config = "E:\mineur\TP6\Codes-20240228\Codes\yolov3-tiny_obj.cfg"
class_file = "obj.names"  # Fichier contenant les noms des classes
net_yolo = load_model(model_config, model_weights)
# -----------------------------------------------
# Detection de visages par OpenCV DNN
# -----------------------------------------------

def euclidean_distance(a, b):
    assert len(a) == len(b)
    squared_diff = np.sum((a - b)**2)
    distance = np.sqrt(squared_diff)
    return distance

def detect_faces_OpenCV_DNN(net, frame):
    in_width = 96
    in_height = 96
    scale = 1/255
    mean = [0,0,0]
    rgb = False
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scale, (in_width, in_height), mean, rgb, crop=False)
    net.setInput(blob)
    detections = net.forward()
    return detections

# Lecture du réseau OpenCV DNN pour la reconnaissance faciale
file_openface = "Codes\openface.nn4.small2.v1.t7"
net_openface = cv2.dnn.readNetFromTorch(file_openface)

# Chemin vers le dossier contenant les images de visages pour la reconnaissance
folder_path = "Codes\\visages"

# Chargement des vecteurs de caractéristiques des visages
feature_vectors = []
file_names = []

for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        features = detect_faces_OpenCV_DNN(net_openface, image)
        feature_vectors.append(features)
        file_names.append(os.path.splitext(filename)[0])  # Enlever l'extension du fichier

feature_vectors = np.array(feature_vectors)

# Lecture de la vidéo en direct
video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()

    # Détection d'objets dans le cadre actuel avec YOLOv3
    detected_frame,boxes = detect_faces_YOLOv3(net_yolo, frame)


    # Extraction des caractéristiques faciales de la frame
    frame_features = detect_faces_OpenCV_DNN(net_openface, detected_frame)


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


        if min_distance < 0.97:  # Définir SEUIL_RECONNAISSANCE selon vos besoins
            closest_name = file_names[closest_match_index]
        else:
            closest_name = "Inconnu"

         # Encadrement du visage avec le nom
        if boxes:
            (x, y, w, h) = (boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = closest_name
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Affichage de la frame de la caméra
    cv2.imshow('Detection de visages par OpenCV DNN', frame)

    # Sortie de la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
video_capture.release()
cv2.destroyAllWindows()
