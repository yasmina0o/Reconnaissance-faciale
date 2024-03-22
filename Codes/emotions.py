from __future__ import print_function
import cv2 as cv
import argparse
import cv2
import numpy as np


in_width = 64       # Resized image width passed to network 64
in_height = 64      # Resized image height passed to network  64
scale = 1.0     # Value scaling factor applied to input pixels 1.0
mean = [127,127,127]    # Mean BGR value subtracted from input image 127 127 127
rgb = False

net = cv2.dnn.readNet("Codes\emotion-ferplus-8.onnx")

classes = [ "neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt" ]

def detectFace(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    # -- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    faceROI = np.array([])
    for (x, y, w, h) in faces:
        faceROI = frame[y:y + h, x:x + w]
        detectAndDisplay(faceROI)
        cv.imshow('Capture - Emotion detection', frame)
    
    return frame
          
        
def detectAndDisplay(frame):
    frameWidth = frame.shape[1]
    mid = int((frameWidth - 110) / 2) + 110 
    leng = frameWidth - mid - 6             
    maxconf = 999
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blob = cv.dnn.blobFromImage(frame_gray, scale, (in_width, in_height), mean, rgb, crop=True)
    net.setInput(blob)
    out = net.forward()
    
    out = out.flatten()
    tabEmotion = []
    for i in range(8):
        conf = out[i] * 100
        if conf > maxconf: conf = maxconf
        if conf < -maxconf: conf = -maxconf
        rlabel = conf*leng/maxconf + mid
        tabEmotion.append(rlabel)

    indice = np.argmax(tabEmotion)
    cv.putText(frame, classes[indice], (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    


parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='Codes\haarcascade_frontalface_default.xml')

parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade

face_cascade = cv.CascadeClassifier()

# -- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)

camera_device = args.camera
# -- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectFace(frame)
    if cv.waitKey(10) == 27:
        break