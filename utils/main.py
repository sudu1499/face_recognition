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
config=yaml.safe_load(open('utils/config.yaml','r'))

resize_to(config) #pre process the images for appropriate shape

create_aug_images(config)
x_train,x_val,x_test,y_train,y_val,y_test=get_x_get_y(config,1)

clb=EarlyStopping(patience=5,restore_best_weights=True)

model=get_model(config)

model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=15,callbacks=[clb])

