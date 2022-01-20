
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception
import yaml
def get_model(config):
    op=config['no_student']
    xmodel=Xception(include_top=False,weights='imagenet',input_shape=(160,160,3))
    
    for i in xmodel.layers:
        i.trainable=False

    model=tf.keras.models.Sequential()
    model.add(xmodel)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(300,activation='relu'))
    model.add(tf.keras.layers.Dense(30,activation='relu'))
    model.add(tf.keras.layers.Dense(op,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return model
def get_model2():
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(20,activation='relu'))
    model.add(tf.keras.layers.Dense(10,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

