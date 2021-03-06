
import cv2
from get_face import get_face
import glob
import os
#this function is used if single photo is given insted of live face detection

def save_detected__faces(config):
    path='/home/sudarshan/Desktop/face_Recog/images/'
    path2='/home/sudarshan/Desktop/face_Recog/images/detected'
    path=config['image_path']
    path2=config['detected_path']
    os.makedirs(path2,exist_ok=True)
    for i in glob.glob(path+"/*"):
        name=i.split('/')[-1]
        c=0
        for j in glob.glob(i+'/*'):
            c+=1
            os.mkdir(path2+name)
            cv2.imwrite(path2+name+'/'+str(c)+'.jpeg',get_face(cv2.imread(j,1)))

