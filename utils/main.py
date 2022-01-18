import yaml
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from get_x_get_y import get_x_get_y
from model import get_model
from tensorflow.keras.callbacks import EarlyStopping

config=yaml.safe_load(open('config.yaml','r'))

x_train,x_val,x_test,y_train,y_val,y_test=get_x_get_y(config)

clb=EarlyStopping(patience=5,restore_best_weights=True)
model=get_model(config)

model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=100,callbacks=[clb])