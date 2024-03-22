import cv2
import numpy as np

# -----------------------------------------------
# Detection de visages par OpenCV DNN avec YOLOv3
# -----------------------------------------------



def load_model(config_path, weight_path):
    return cv2.dnn.readNetFromDarknet(config_path, weight_path)

def detect_faces_YOLOv3(net, frame, min_conf=0.5, nms_threshold=0.4):
    # Récupération des dimensions de l'image
    (H, W) = frame.shape[:2]

    # Construction d'un blob à partir de l'image d'entrée et mise à jour du réseau avec ce blob
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Passage de l'image à travers le réseau et récupération des détections et des confiances
    outputs = net.forward(output_layers)

    # Initialisation des listes pour les boîtes englobantes, confiances et classes
    boxes = []
    confidences = []
    class_ids = []

    # Analyse des détections et sélection des détections de visage
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > min_conf:
                # Récupération des coordonnées de la boîte englobante
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Calcul des coordonnées de la boîte englobante
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Ajout des boîtes englobantes, confiances et identifiants de classe
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Application de la suppression non maximale pour éliminer les boîtes englobantes redondantes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, min_conf, nms_threshold)

    # Vérification des indices sélectionnés après la suppression non maximale
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Tracé des boîtes englobantes et des classes détectées
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "Face"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

# Lecture du modèle YOLOv3
model_weights = "E:\mineur\TP6\Codes-20240228\Codes\yolov3-tiny_obj_best.weights"
model_config = "E:\mineur\TP6\Codes-20240228\Codes\yolov3-tiny_obj.cfg"
class_file = "obj.names"  # Fichier contenant les noms des classes
net = load_model(model_config, model_weights)



# boucle de lecture/traitement
video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()

    # Détection d'objets dans le cadre actuel
    detected_frame = detect_faces_YOLOv3(net, frame)

    # Affichage du cadre avec les détections
    cv2.imshow('Object Detection', detected_frame)

    # Interruption si la touche 'q' est enfoncée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Arrêt de la capture vidéo et fermeture de toutes les fenêtres
video_capture.release()
cv2.destroyAllWindows()
