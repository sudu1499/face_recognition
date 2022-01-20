import cv2
from glob import glob

def resize_to(config):
    path=config['image_path']

    for i in glob(path+'/*'):
        name=i.split('/')[-1]
        for j in glob(i+'/*'):
            f=j.split('/')[-1]
            img=cv2.imread(j,1)
            img=cv2.resize(img,(160,160))
            cv2.imwrite(j,img)
