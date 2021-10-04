import os
import numpy as np  
import cv2 as cv 
import sys


file = sys.argv[1]
print(file)

haar_cascade = cv.CascadeClassifier('data/haar_face.xml')
#features = np.load('features.npy')
#labels = np.load('labels.npy')


face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

dir = 'photos/faces'
people = []
for f in os.listdir(dir):
    people.append(f)

img = cv.imread(file)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("person", gray)

faces_rect = haar_cascade.detectMultiScale(gray, 1.1 ,4)


for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]}  with confidence of {confidence}')
    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255.0), thickness=2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), thickness=2)


cv.imshow("detected ", img)
cv.waitKey(0)