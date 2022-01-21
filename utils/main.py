import yaml
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from utils.get_x_get_y import get_x_get_y
from utils.model import get_model
from tensorflow.keras.callbacks import EarlyStopping
from utils.make_augmented_images import create_aug_images
from glob import glob
from tensorflow.keras.applications.xception import Xception
from utils.pre_process import resize_to
from sklearn.metrics import accuracy_score
config=yaml.safe_load(open('utils/config.yaml','r'))


create_aug_images(config) #augment the data

resize_to(config) #pre process the images for appropriate shape

x_train,x_val,x_test,y_train,y_val,y_test=get_x_get_y(config,1)

clb=EarlyStopping(patience=5,restore_best_weights=True)

model=get_model(config,x_train,x_val,y_train,y_val,clb)

p=model.predict(x_test)
print(accuracy_score(np.argmax(y_test,axis=1),np.argmax(p,axis=1)))
