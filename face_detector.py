import os
import numpy as np
import cv2
from keras.models import load_model


input_shape = (120,120,3)
labels_dict = {0: 'WithMask', 1: 'WithoutMask'}
color_dict = {0 : (0,255,0), 1:(0,0,255)}
model = load_model('best_model.hdf5')

from mtcnn.mtcnn import MTCNN 
detector = MTCNN() 

size = 4
webcam = cv2.VideoCapture(0)

while True: 
    (rval, im) = webcam.read()
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))
    rgb_image = cv2.cvtColor(mini, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(mini)


    for f in faces:
        x, y, w, h = [v * size for v in f['box']]

        face_img = im[y:y + h, x:x + w]
        print(face_img)
        resized = cv2.resize(face_img, (input_shape[0],input_shape[1]))

        reshaped = np.reshape(resized, (1, input_shape[0],input_shape[1], 3)) 

        result = model.predict(reshaped)
        print(result)

        label = np.argmax(result, axis=1)[0] 

        cv2.rectangle(im, (x, y), (x + w, y + h), color_dict[label], 2) 
        cv2.rectangle(im, (x, y - 40), (x + w, y), color_dict[label], -1) 
        cv2.putText(im, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('LIVE FACE DETECTION', im)
    key = cv2.waitKey(10)
    if key == 27: 
        break
webcam.release()
cv2.destroyAllWindows()
