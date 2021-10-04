import os
import cv2 as cv 
import numpy as np

dir = 'photos/faces'

people = []
for f in os.listdir(dir):
    people.append(f)

haar_cascade = cv.CascadeClassifier('data/haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(dir, person)
        label = people.index(person)
        print(f'indexing: {path}')
        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            # haar_cascade - is very sesitive to noise
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)



create_train()

print(f'features list: { len(features) }')
print(f'features labels: { len(labels) }')


features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)






