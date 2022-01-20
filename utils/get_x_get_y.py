
import cv2
from glob import glob
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

#to get x as image and y as encoded value
def get_x_get_y(config,flag=1): #here flag is wether u want y
    #path should contain detected face directory without / at the end
    x=[]
    y=[]
    path=config['image_path']
    enc=OneHotEncoder(sparse=True)
    for i in glob(path+'/*'):
        name=i.split('/')[-1]
        for j in glob(i+'/*'):
            im=cv2.imread(j,1)
            im=cv2.resize(im,((160,160)))
            im=np.reshape(im,(160,160,3))
            x.append(im)
            y.append(name)
    x=np.array(x)
    if flag==0:
        return x
    y=np.array(y)
    y=y.reshape((-1,1))
    y=enc.fit_transform(y).toarray()
    x_,x_test,y_,y_test=train_test_split(x,y,test_size=.2,random_state=10)
    x_train,x_val,y_train,y_val=train_test_split(x_,y_,test_size=.2,random_state=10)
    return x_train,x_val,x_test,y_train,y_val,y_test

