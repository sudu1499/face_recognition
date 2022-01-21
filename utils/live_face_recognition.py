import cv2
from cv2 import VideoCapture
from tensorflow.keras.models import load_model
import dlib
import numpy as np
model=load_model('./utils/saved_models/model1.pb')
det=dlib.get_frontal_face_detector()
vid=cv2.VideoCapture(0)
while 1:
    _,frame=vid.read()
    frame1=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    f=det(frame1)
    for i in f:
        croped=frame[i.top():i.bottom(),i.left():i.right()]
        cv2.imshow('',croped)
        croped=cv2.resize(croped,(160,160))
        croped=croped.reshape((1,160,160,3))
        print(np.argmax(model.predict(croped/255),axis=1))
    if cv2.waitKey(1)==ord('q'):
        cv2.destroyAllWindows()
        vid.release()
        break

