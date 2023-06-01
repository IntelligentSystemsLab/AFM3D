# -*- coding: utf-8 -*-
"""
@author: ls
"""

import pandas as pd
from PIL import Image
from torchvision import transforms as T
import os
import numpy as np
import random
import torch

train_data_list = pd.read_csv(
    'AUC_Distracted_Driver_Dataset/auc.distracted.driver.dataset_v2/v1_cam1_no_split/Train_data_list.csv',na_values='na')
x_path = [x[19:] for x in train_data_list['Image']]
x_path = [('AUC_Distracted_Driver_Dataset/auc.distracted.driver.dataset_v2/v1_cam1_no_split/'+x) for x in x_path]
y_list = train_data_list.iloc[:,1]

x_train = []
y_train = []#train_data_list.iloc[:,1]
cnt = 0
for img_name,label_now in zip(x_path,y_list):
    transforms = T.Compose([T.RandomResizedCrop(224), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    if os.path.exists(img_name)==False:
        continue
    else:
        img = Image.open(img_name)
        img = transforms(img)
        x_train.append(img)
        y_train.append(label_now)
        cnt+=1
        if (cnt%500)==0:
            print('processing: ',cnt)

num_classes = 10
num_clients = 15
alpha = 0.5

def dirichlet_split_noniid(alpha, num_classes, num_clients):
    label_distribution = np.random.dirichlet([alpha]*num_classes, num_clients)
    return label_distribution

classes_weight_all = dirichlet_split_noniid(alpha,num_classes,num_clients)

idx_total = []

for i in range(0,len(y_train)):
    idx_total.append(i)        

#x_train = np.array(x_train)
y_train = np.array(y_train)
for j in range(1,num_clients + 1):###############################################################
    classes = [0,1,2,3,4,5,6,7,8,9]
    
    Smin, Smax =1000,1200 #the range of the local train data size
    num = random.randint(Smin,Smax) #the number of local train data size
    Pclasses = (classes_weight_all[j-1]*num).round() #the train number of each classes    
   
    Pclasses = Pclasses.astype('int')
    idx_local = []
    
    for i in range(num_classes):
        index_range =  np.argwhere(y_train == classes[i])
        index_max = max(index_range)
        index_min = min(index_range)
        idx_local = idx_local+random.sample(
            idx_total[int(index_min):int(index_max)],min((int(index_max-index_min)),Pclasses[classes[i]]) )

    x_train_local = []
    for idx_now in idx_local:
        x_train_local.append(x_train[idx_now])
    y_train_local = y_train[idx_local]
    #y_train_local = torch.tensor(y_train_local)
    train_image_local = [t for t in zip(x_train_local,y_train_local)]
    torch.save(train_image_local, 'AUC_Distracted_Driver_Dataset/auc.distracted.driver.dataset_v2/v1_cam1_no_split/data_client/train_client/'+str(j)+'.pt') # 保存
#'''    
