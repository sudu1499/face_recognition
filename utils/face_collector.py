from lib2to3.pgen2.token import SLASH
import cv2
import dlib
import os
import yaml
import os
import platform
if platform.system()=='Windows':
    slash='\\'
else:
    slash='/'
config_path=os.path.dirname(os.path.abspath(__file__))
config_file=config_path+slash+'config.yaml'
config=yaml.safe_load(open(config_file,'r'))

#path='/home/sudarshan/Desktop/face_Recog/images/'
path=config['image_path']
no_students=config['no_student']
vid=cv2.VideoCapture(0)
name=input("your name")
os.makedirs(path+name,exist_ok=True)
no_students+=1
config['no_student']=no_students
with open(config_file,'w') as f:
    f.write(yaml.safe_dump(config))
det=dlib.get_frontal_face_detector()
j=0
count=0
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
        count+=1
    if  count==100:
        break
    cv2.imshow('',frame)
    if cv2.waitKey(1)==ord('q'):
        cv2.destroyAllWindows()
        break
vid.release()
