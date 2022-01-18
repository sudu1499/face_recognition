import cv2
import dlib
import glob
import os

path='/home/sudarshan/Desktop/face_Recog/images/'
vid=cv2.VideoCapture(0)
name=input("your name")
os.makedirs(path+name,exist_ok=True)
det=dlib.get_frontal_face_detector()
j=0
while 1:
    j+=1
    _,frame=vid.read()
    f2=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=det(f2)
    for i in face:
        x1,y1,x2,y2=i.left(),i.top(),i.right(),i.bottom()
        croped=frame[y1:y2,x1:x2]
        cv2.imwrite(path+name+'/'+name+str(j)+'.jpeg',croped) 
        cv2.rectangle(frame,(x1,y1),(x2,y2),color=(255,0,0),thickness=3)
    cv2.imshow('',frame)
    if cv2.waitKey(1)==ord('q'):
        cv2.destroyAllWindows()
        break

vid.release()
