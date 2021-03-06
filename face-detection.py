import cv2 
import argparse
import numpy as np
from fastai.vision.all import *
from PIL import Image

## Cascade ref:
## https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

## FastAI ref:
## https://github.com/fastai/fastbook/blob/master/02_production.ipynb

def is_obama(x): 
    return 'obama' in x

global learn
learn = load_learner('models/export.pkl')

global img1
global img2

img1 = cv2.namedWindow("img1", cv2.WINDOW_NORMAL)
img2 = cv2.namedWindow("img2", cv2.WINDOW_NORMAL)


def detect_faces(frame):
    frame_gray = cv2.equalizeHist(frame)
    faces = face_cascade.detectMultiScale(frame_gray)
    return faces

def classify(faces, frame):
    faceROI = None
    face_class = None
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        
        #bounding box dos rostos
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        faceROI = frame[y:y+h,x:x+w]
        #print(faceROI)
        
        # # Acha os olhos
        # eyes = eyes_cascade.detectMultiScale(faceROI)
        # for (x2,y2,w2,h2) in eyes:
        #     #bounding box dos olhos
        #     frame = cv2.rectangle(frame, (x + x2, y + y2), (x + x2 + w2, y + y2+h2), (0, 0, 255), 2)

    
    if(faceROI is not None):
        cv2.imshow('img2',faceROI)
        t = torch.tensor(frame)
        p, tensor, probs = learn.predict(t)

        if(str(p) == 'False'):
            face_class = 'chris'
        else:
            face_class = 'obama'

        frame = cv2.putText(frame, face_class,(x+w, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0),3)

        #print(face_class)

        cv2.imshow('img1', frame)

### configura -- 1. Carrega cascades
parser = argparse.ArgumentParser(description='Cascade Classifier')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='models/haarcascades/haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='models/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()


if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
camera_device = args.camera

# Captura o video
vid = cv2.VideoCapture(camera_device) 

while(True): 
      
    # Captura cada frame do video
    ret, frame = vid.read() 

    # transforma em cinza para facilitar a busca pelo  padr??o
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Acha os rostos
    faces = detect_faces(gray)

    #mostra bounding boxes
    classify(faces,frame)

    # Mostra o frame atual, pode ou n??o estar com as bordas coloridas
    #cv2.imshow('img1', frame) 

    # Bot??o q para iniciar a calibra????o e depois sair do programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Solta o objeto de captura
vid.release() 

# Destroi as janelas
cv2.destroyAllWindows() 
