import cv2

def detect_faces_OpenCV_DNN(net, frame, min_conf=0.5):
    in_width = 160      # Resized image width passed to network
    in_height = 120     # Resized image height passed to network
    scale = 1.0      # Value scaling factor applied to input pixels
    mean = [104.0, 177.0, 123.0] # Mean BGR value subtracted from input image
    rgb = False          # True if model expects RGB inputs, otherwise it expects BGR

    h = frame.shape[0]
    w = frame.shape[1]

    # Mise au format blob pour le CNN
    blob = cv2.dnn.blobFromImage(frame, scale, (in_width, in_height), mean, rgb, crop=False)

    # Entrée du réseau
    net.setInput(blob)
    # calcul de la sortie
    detections = net.forward()

    bboxes = [] # liste des boites englobantes (bounding boxes) contenant des visages
    for i in range(detections.shape[2]):
        # seuls les visages qui sont suffisament sûrs sont conservés
        confidence = detections[0, 0, i, 2]
        if confidence > min_conf:
            x1 , y1 = int(detections[0, 0, i, 3] * w), int(detections[0, 0, i, 4] * h)
            x2 , y2  = int(detections[0, 0, i, 5] * w),int( detections[0, 0, i, 6] * h)
            bboxes.append([x1, y1, x2, y2])
            # tracé des bb
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), h // 150, 8,)
    return frame, bboxes


# Lecture du réseau
# pour d'autres réseaux voir http://jevois.org/basedoc/PyDetectionDNN_8py_source.html
# pensez à changer le fichier contenant le modele et le fichier contenant la configuration
model_file = "res10_300x300_ssd_iter_140000_fp16.caffemodel" # modele
config_file = "deploy.prototxt" # configuration
backend = cv2.dnn.DNN_BACKEND_OPENCV
target = cv2.dnn.DNN_TARGET_CPU
net = cv2.dnn.readNet(config_file, model_file)
net.setPreferableBackend(backend)
net.setPreferableTarget(target)


# boucle de lecture/traitement
video_capture = cv2.VideoCapture(4)
while True:

    _, frame = video_capture.read()

    frame, bboxes = detect_faces_OpenCV_DNN(net, frame)

    cv2.imshow('Detection de visages par openCV DNN', frame)


    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

video_capture.release()
cv2.destroyAllWindows()
