import cv2 
import argparse
import numpy as np

## Ref:
## https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

def detect_faces(frame):
    frame_gray = cv2.equalizeHist(frame)
    faces = face_cascade.detectMultiScale(frame_gray)
    return faces

def display_boxes(faces, frame):
    faceROI = None
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        
        #bounding box dos rostos
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        faceROI = frame[y:y+h,x:x+w]
        #print(faceROI)
        
        # Acha os olhos
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            #bounding box dos olhos
            frame = cv2.rectangle(frame, (x + x2, y + y2), (x + x2 + w2, y + y2+h2), (0, 0, 255), 2)

    cv2.imshow('img1', frame)
    if(faceROI is not None):
        cv2.imshow('img2',faceROI)

### configura -- 1. Carrega cascades
parser = argparse.ArgumentParser(description='Cascade Classifier')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/haarcascades/haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
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

cv2.namedWindow("img1", cv2.WINDOW_NORMAL)
cv2.namedWindow("img2", cv2.WINDOW_NORMAL)

while(True): 
      
    # Captura cada frame do video
    ret, frame = vid.read() 

    # transforma em cinza para facilitar a busca pelo  padrão
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Acha os rostos
    faces = detect_faces(gray)

    #mostra bounding boxes
    display_boxes(faces,frame)

    # Mostra o frame atual, pode ou não estar com as bordas coloridas
    #cv2.imshow('img1', frame) 

    # Botão q para iniciar a calibração e depois sair do programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Solta o objeto de captura
vid.release() 

# Destroi as janelas
cv2.destroyAllWindows() 
