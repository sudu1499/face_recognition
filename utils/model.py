
from io import open_code
import tensorflow as tf
from tensorflow.keras.applications import xception,vgg16
import yaml
def get_model(config,x_train,x_val,y_train,y_val,clb):
    op=config['no_student']
    print('number of students are',op)
    #xmodel=xception.Xception(include_top=False,weights='imagenet',input_shape=(270,480,3))
    vgg=vgg16.VGG16(include_top=False,weights='imagenet',input_shape=(224,224,3))
    print(x_train.shape)
    for i in vgg.layers:
        i.trainable=False

    model=tf.keras.models.Sequential()
    model.add(vgg)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(200,activation='relu'))
    model.add(tf.keras.layers.Dense(30,activation='relu'))
    model.add(tf.keras.layers.Dense(op,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    print(model.summary())
    model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=15,batch_size=16,callbacks=[clb])
    return model


def get_model2():
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(20,activation='relu'))
    model.add(tf.keras.layers.Dense(10,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

