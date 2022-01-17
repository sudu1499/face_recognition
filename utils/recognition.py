from  tensorflow.keras.models import load_model
import cv2
import get_face
import numpy as np

vid=cv2.VideoCapture(0)
while 1:
    _,frame=vid.read()
    frame=get_face(frame)
    model=load_model()
    p=model.predict(frame)
    print(np.argmax(p))

    