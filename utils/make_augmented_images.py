from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import numpy as np
import cv2


def create_aug_images(config): #does augmentation for all images in the same folder
    path=config['image_path']
    datagen=ImageDataGenerator(width_shift_range=.2,height_shift_range=.2,zoom_range=.2,rotation_range=.2)

    for i in glob(path+'/*'):
        name=i.split('/')[-1]
        for j in glob(i+'/*'):
            img=cv2.imread(j,1)
            img=np.reshape(img,((1,)+img.shapes))
            c=0
            for d in datagen.flow(img,batch_size=1,save_to_dir=path+'/'+name+'/',save_format='.jpeg'):
                c+=1
                if c==5:
                    break

