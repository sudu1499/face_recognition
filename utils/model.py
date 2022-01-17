import tensorflow as tf
import glob
import cv2
from sklearn.model_selection import train_test_split

path='/home/sudarshan/Desktop/face_Recog/images/'
y=[]
temp=[]
x=[]
for i in glob.glob(path+'*'):
    y.append(i.split('/')[-1])
for i in glob.glob(path+"*/**"):
    temp=cv2.imread(i,1)
    x.append(temp)

