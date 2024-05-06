import os
import cv2 as cv
import numpy as np

p = []
for i in os.listdir('C:/Users/qasim/OneDrive/Desktop/Face Recognition with OpenCV/Faces/train'):
    p.append(i)
DIR = r'C:/Users/qasim/OneDrive/Desktop/Face Recognition with OpenCV/Faces/train'

haar_cascasde = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []
def create_train():
    for person in p:
        path = os.path.join(DIR, person)
        label = p.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascasde.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer.create()

face_recognizer.train(features, labels)
face_recognizer.save('face_trained.yml')

np.save("features.npy", features)
np.save("labels.npy", labels)