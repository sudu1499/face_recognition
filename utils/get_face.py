import cv2
import dlib

def get_face(im1):
    croped=[]
    det=dlib.get_frontal_face_detector()

    f2=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    face=det(f2)
    for p in face:
        x1,y1,x2,y2=p.left(),p.top(),p.right(),p.bottom()
        croped.append(im1[y1:y2,x1:x2])
    return croped
