import cv2
from utils import get_face
import glob
import os


def save_faces():
    path='/home/sudarshan/Desktop/face_Recog/images/'
    path2='/home/sudarshan/Desktop/face_Recog/images/detected'
    os.makedirs(path2,exist_ok=True)
    for i in glob.glob(path+"*"):
        name=i.split('/')[-1]
        c=0
        for j in glob.glob(i+'/*'):
            c+=1
            os.mkdir(path2+name)
            cv2.imwrite(path2+name+'/'+str(c)+'.jpeg',get_face(cv2.imread(j,1)))

