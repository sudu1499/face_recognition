import tensorflow as tf
import glob
import cv2
from get_splited_data import get_splited_data
from tensorflow.keras.applications import Facenet
def create_model(config):
    path=config['image_path']  #--->this path should be image path
    x_train,x_val,x_test,y_train,y_val,y_test=get_splited_data(path)

    



