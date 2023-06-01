import torch
from torchvision  import models
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
import os
from client_AFM3DAver_SFD_densenet121 import Client
import random

class Metanet(nn.Module):
  def __init__(self, device,local_metatrain_epoch=1, local_test_epoch=3,outer_lr=0.0001,inner_lr=0.0001):
    super(Metanet, self).__init__()
    self.device = device
    self.local_metatrain_epoch = local_metatrain_epoch
    self.local_test_epoch = local_test_epoch
    self.net = models.densenet121(pretrained=False).to(self.device)
    self.loss_function = torch.nn.CrossEntropyLoss()
    self.clients = []
    self.test_clients = []
    self.mode_1 = "fomaml_train"
    self.mode_2 = "fomaml_test"
    self.batch_size = 40
    self.path_now = os.path.dirname(__file__)
    self.last_path = '/0529_1_AFM3DAver_SFD_densenet121'#-----------------------------------------------------
    train_path = self.path_now + r"/state-farm-distracted-driver-detection/data_client/train_client"
    test_path = self.path_now + r"/state-farm-distracted-driver-detection/data_client/test_client"
    train_file_set = os.listdir(train_path)
    train_path_set = [os.path.join(train_path,i) for i in train_file_set]
    test_file_set = os.listdir(test_path)
    test_path_set = [os.path.join(test_path,i) for i in test_file_set]
    self.time_accum = [0]
    
    for index,path in enumerate(train_path_set):
      model = models.densenet121(pretrained=False).to(self.device)
      self.clients.append(Client(model,index,path,'train',local_metatrain_epoch,local_test_epoch,
                                 inner_lr,outer_lr,self.device,self.mode_1,self.batch_size))
    for index,path in enumerate(test_path_set):
      model = models.densenet121(pretrained=False).to(self.device)
      self.test_clients.append(Client(model,index,path,'test',local_metatrain_epoch,local_test_epoch,
                                inner_lr,outer_lr,self.device,self.mode_2,self.batch_size))

  def forward(self):
    pass

  def save_time(self,save_path):
    dataframe = pd.DataFrame(list(self.time_accum), columns=['time_accum'])
    dataframe.to_excel(save_path, index=False)
  
  def meta_training(self,round):
    
    id_train_0 = list(range(len(self.clients)))
    for id,j in enumerate(id_train_0):
      if self.clients[j].time <= 0:
        self.clients[j].refresh(self.net)
        self.clients[j].local_asymeta_fomaml_train()
        self.clients[j].epoch = round
      else:
        continue
    id_train = []
    size_all = 0
    for id in id_train_0:
      #self.clients[id].time = max(self.clients[id].time - 40,0)
      if round<=1:#---------------------------------------------------------------------
        self.clients[id].time = max(self.clients[id].time - 10,0)
      else:
        self.clients[id].time = max(self.clients[id].time - 5,0)
      if self.clients[id].time <= 0:
        id_train.append(id)
        size_all += self.clients[id].size
    
    weight = []
    for id,j in enumerate(id_train):
      #weight.append(self.clients[j].size / size_all)
      weight.append(1.0)
    weight = np.array(weight)
    weight = weight / weight.sum()

    for id,j in enumerate(id_train):
      for w,w_t in zip(self.net.parameters(),self.clients[j].net.parameters()):
        if (w is None or id == 0):
          w_tem = Variable(torch.zeros_like(w)).to(self.device)
          w.data.copy_(w_tem.data)
        if w_t is None:
          w_t = Variable(torch.zeros_like(w)).to(self.device)
        
        w.data.add_(w_t.data*weight[id])

  def Testing(self,round):
    id_test = list(range(len(self.test_clients)))
    final_init_loss, final_init_acc, final_init_pre\
        , final_init_recall, final_init_f1 = [], [], [], [], []
    for a,id in enumerate(id_test):
      self.test_clients[id].refresh(self.net)
      fine_tuning_loss_list,test_loss_list,init_loss\
        ,fine_tuning_acc_list,test_acc_list,init_acc\
        ,fine_tuning_pre_list,test_pre_list,init_pre\
        ,fine_tuning_recall_list,test_recall_list,init_recall\
        ,fine_tuning_f1_list,test_f1_list,init_f1 = self.test_clients[id].test()
      if a == 0:
        final_train_loss = fine_tuning_loss_list.copy()
        final_test_loss = test_loss_list.copy()
        final_init_loss.append(init_loss)
        
        final_train_acc = fine_tuning_acc_list.copy()
        final_test_acc = test_acc_list.copy()
        final_init_acc.append(init_acc)
        
        final_train_pre = fine_tuning_pre_list.copy()
        final_test_pre = test_pre_list.copy()
        final_init_pre.append(init_pre)
                
        final_train_recall = fine_tuning_recall_list.copy()
        final_test_recall = test_recall_list.copy()
        final_init_recall.append(init_recall)
                
        final_train_f1 = fine_tuning_f1_list.copy()
        final_test_f1 = test_f1_list.copy()
        final_init_f1.append(init_f1)

      else:
        final_train_loss = np.concatenate((final_train_loss,fine_tuning_loss_list),axis = 0)
        final_test_loss = np.concatenate((final_test_loss,test_loss_list),axis = 0)
        final_init_loss.append(init_loss)
        
        final_train_acc = np.concatenate((final_train_acc,fine_tuning_acc_list),axis = 0)
        final_test_acc = np.concatenate((final_test_acc,test_acc_list),axis = 0)
        final_init_acc.append(init_acc)
        
        #---
        final_train_pre = np.concatenate((final_train_pre,fine_tuning_pre_list),axis = 0)
        final_test_pre = np.concatenate((final_test_pre,test_pre_list),axis = 0)
        final_init_pre.append(init_pre)
        #---
        final_train_recall = np.concatenate((final_train_recall,fine_tuning_recall_list),axis = 0)
        final_test_recall = np.concatenate((final_test_recall,test_recall_list),axis = 0)
        final_init_recall.append(init_recall)
        #---
        final_train_f1 = np.concatenate((final_train_f1,fine_tuning_f1_list),axis = 0)
        final_test_f1 = np.concatenate((final_test_f1,test_f1_list),axis = 0)
        final_init_f1.append(init_f1)
    
    last_path = self.last_path
    loss_train_file_path = self.path_now + "/result_save/farm/loss/async_fomaml_train_loss" + last_path
    loss_test_file_path = self.path_now + "/result_save/farm/loss/async_fomaml_test_loss" + last_path
    loss_init_file_path = self.path_now + "/result_save/farm/loss/async_fomaml_init_loss" + last_path
    
    acc_train_file_path = self.path_now + "/result_save/farm/acc/async_fomaml_train_acc" + last_path
    acc_test_file_path = self.path_now + "/result_save/farm/acc/async_fomaml_test_acc" + last_path
    acc_init_file_path = self.path_now + "/result_save/farm/acc/async_fomaml_init_acc" + last_path
    
    #---
    pre_train_file_path = self.path_now + "/result_save/farm/pre/async_fomaml_train_pre" + self.last_path
    pre_test_file_path = self.path_now + "/result_save/farm/pre/async_fomaml_test_pre" + self.last_path
    pre_init_file_path = self.path_now + "/result_save/farm/pre/async_fomaml_init_pre" + self.last_path
    #---
    recall_train_file_path = self.path_now + "/result_save/farm/recall/async_fomaml_train_recall" + self.last_path
    recall_test_file_path = self.path_now + "/result_save/farm/recall/async_fomaml_test_recall" + self.last_path
    recall_init_file_path = self.path_now + "/result_save/farm/recall/async_fomaml_init_recall" + self.last_path
    #---
    f1_train_file_path = self.path_now + "/result_save/farm/f1/async_fomaml_train_f1" + self.last_path
    f1_test_file_path = self.path_now + "/result_save/farm/f1/async_fomaml_test_f1" + self.last_path
    f1_init_file_path = self.path_now + "/result_save/farm/f1/async_fomaml_init_f1" + self.last_path
    
    if not os.path.exists(loss_train_file_path):
      os.mkdir(loss_train_file_path)
    if not os.path.exists(loss_test_file_path):
      os.mkdir(loss_test_file_path)
    if not os.path.exists(loss_init_file_path):
      os.mkdir(loss_init_file_path)
      
    if not os.path.exists(acc_train_file_path):
      os.mkdir(acc_train_file_path)
    if not os.path.exists(acc_test_file_path):
      os.mkdir(acc_test_file_path)
    if not os.path.exists(acc_init_file_path):
      os.mkdir(acc_init_file_path)
    #---
    if not os.path.exists(pre_train_file_path):
      os.mkdir(pre_train_file_path)
    if not os.path.exists(pre_test_file_path):
      os.mkdir(pre_test_file_path)
    if not os.path.exists(pre_init_file_path):
      os.mkdir(pre_init_file_path)
    #---
    if not os.path.exists(recall_train_file_path):
      os.mkdir(recall_train_file_path)
    if not os.path.exists(recall_test_file_path):
      os.mkdir(recall_test_file_path)
    if not os.path.exists(recall_init_file_path):
      os.mkdir(recall_init_file_path)
    #---
    if not os.path.exists(f1_train_file_path):
      os.mkdir(f1_train_file_path)
    if not os.path.exists(f1_test_file_path):
      os.mkdir(f1_test_file_path)
    if not os.path.exists(f1_init_file_path):
      os.mkdir(f1_init_file_path)    

    loss_train_path = os.path.join(loss_train_file_path,"{}.npy".format(round))
    loss_test_path = os.path.join(loss_test_file_path,"{}.npy".format(round))
    loss_init_path = os.path.join(loss_init_file_path,"{}.npy".format(round))
    
    acc_train_path = os.path.join(acc_train_file_path,"{}.npy".format(round))
    acc_test_path = os.path.join(acc_test_file_path,"{}.npy".format(round))
    acc_init_path = os.path.join(acc_init_file_path,"{}.npy".format(round))
    
    #---
    pre_train_path = os.path.join(pre_train_file_path,"{}.npy".format(round))
    pre_test_path = os.path.join(pre_test_file_path,"{}.npy".format(round))
    pre_init_path = os.path.join(pre_init_file_path,"{}.npy".format(round))
    #---
    recall_train_path = os.path.join(recall_train_file_path,"{}.npy".format(round))
    recall_test_path = os.path.join(recall_test_file_path,"{}.npy".format(round))
    recall_init_path = os.path.join(recall_init_file_path,"{}.npy".format(round))
    #---
    f1_train_path = os.path.join(f1_train_file_path,"{}.npy".format(round))
    f1_test_path = os.path.join(f1_test_file_path,"{}.npy".format(round))
    f1_init_path = os.path.join(f1_init_file_path,"{}.npy".format(round))

    np.save(loss_train_path,final_train_loss)
    np.save(loss_test_path,final_test_loss) 
    np.save(loss_init_path,final_init_loss) 
    
    np.save(acc_train_path,final_train_acc)
    np.save(acc_test_path,final_test_acc) 
    np.save(acc_init_path,final_init_acc)
    
    #---
    np.save(pre_train_path,final_train_pre)
    np.save(pre_test_path,final_test_pre) 
    np.save(pre_init_path,final_init_pre) 
    #---
    np.save(recall_train_path,final_train_recall)
    np.save(recall_test_path,final_test_recall) 
    np.save(recall_init_path,final_init_recall) 
    #---
    np.save(f1_train_path,final_train_f1)
    np.save(f1_test_path,final_test_f1) 
    np.save(f1_init_path,final_init_f1) 

