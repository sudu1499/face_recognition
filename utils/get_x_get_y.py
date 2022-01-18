
import cv2
from glob import glob
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
#to get x as image and y as encoded value
def get_x_get_y(path):
    #path should contain detected face directory without / at the end
    x=[]
    y=[]
    enc=OneHotEncoder(sparse=False)
    for i in glob(path+'/*'):
        name=i.split('/')[-1]
        for j in glob(i+'/*'):
            im=cv2.imread(j,1)
            im=cv2.resize(im,((160,160)))
            im=np.reshape(im,(160,160,3))
            x.append(im)
            y.append(name)
    x=np.array(x)
    y=np.array(y)
    y=y.reshape((-1,1))
    y=enc.fit_transform(y)
    return np.array(x),y
path='/home/sudarshan/Desktop/face_Recog/images'
v=get_x_get_y(path)
