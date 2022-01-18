from pickletools import optimize
import weakref
import tensorflow as tf
from tensorflow.keras.applications import xception
def get_model(config):
    op=config['no_student']
    xmodel=xception(include_top=False,weights='imagenet',input_shape=(160,160,3))
    
    for i in xmodel.layers:
        i.trainable=False

    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(300,activation='relu'))
    model.add(tf.keras.layers.Dense(30,activation='relu'))
    model.add(tf.keras.layers.Dense(op,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return model
    



