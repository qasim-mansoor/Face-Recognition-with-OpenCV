import numpy as np
import cv2 as cv
import os

haar_cascasde = cv.CascadeClassifier('haar_face.xml')

p = []
for i in os.listdir('C:/Users/qasim/OneDrive/Desktop/Face Recognition with OpenCV/Faces/train'):
    p.append(i)
# features = np.load('features.npy')
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer.create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'C:/Users/qasim/OneDrive/Desktop/Face Recognition with OpenCV/Faces/val/ben_afflek/5.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detecting the face
faces_rect = haar_cascasde.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    label, conf = face_recognizer.predict(faces_roi)
    print(f'Label = {p[label]} with confidence of {conf}')

    cv.putText(img, str(p[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)

    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

cv.imshow("Detected Face", img)
cv.waitKey(0)