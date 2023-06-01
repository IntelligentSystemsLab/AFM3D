# -*- coding: utf-8 -*-
"""
@author: ls
"""

import os
import random
import numpy as np
import pandas as pd 
import torch
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from dask.array.image import imread
from dask import bag, threaded
from dask.diagnostics import ProgressBar
#import cv2
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import math
from torchvision import transforms as T
from PIL import Image

driver_details = pd.read_csv('state-farm-distracted-driver-detection/driver_imgs_list.csv',na_values='na')
print(driver_details.head(5))

## Getting all the images

train_image = []
image_label = []


for i in range(10):
    print('now we are in the folder C',i)
    imgs = os.listdir("state-farm-distracted-driver-detection/imgs/train/train/c"+str(i))
    for j in range(len(imgs)):
    #for j in range(100):
        img_name = "state-farm-distracted-driver-detection/imgs/train/train/c"+str(i)+"/"+imgs[j]
        #img = cv2.imread(img_name)
        #img = color.rgb2gray(img)
        #img = img[50:,120:-50]
        #img = cv2.resize(img,(224,224))
        transforms = T.Compose([T.RandomResizedCrop(224), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        img = Image.open(img_name)
        img = transforms(img)
        label = i
        driver = driver_details[driver_details['img'] == imgs[j]]['subject'].values[0]
        train_image.append([img,label,driver])
        image_label.append(i)

drive_id_train = ['p039','p081','p047','p024','p049','p035','p042','p045',
                  'p072','p015','p012','p064','p002','p056','p014','p050',
                  'p041','p052','p026','p075','p051']
drive_id_test = ['p016','p066','p061','p021','p022']

## Splitting and saving the train and test cients
for driver_id in  drive_id_train:
    train_image_local = []
    for features,labels,drivers in train_image:
        if drivers == driver_id:
            train_image_local.append([features,labels])
    #torch.save(train_image_local, 'state-farm-distracted-driver-detection/data_client/train_client/'+driver_id+'.pt') # 保存
    
for driver_id in  drive_id_test:
    test_image_local = []
    for features,labels,drivers in train_image:
        if drivers == driver_id:
            test_image_local.append([features,labels])
    #torch.save(test_image_local, 'state-farm-distracted-driver-detection/data_client/test_client/'+driver_id+'.pt') # 保存
