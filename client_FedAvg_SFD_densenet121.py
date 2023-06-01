import torch
import numpy as np
from torch import nn
#from torch.nn import functional as F
from torchvision import datasets, transforms
from copy import deepcopy
from torch.autograd import Variable 
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import time
import random
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import itertools

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.idxs = [int(i) for i in range(len(dataset))]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.as_tensor(image), torch.as_tensor(label)

class Client(nn.Module):
    def __init__(self,model,id,path,type_client,update_step,update_step_test,base_lr,meta_lr,device,mode,batch_size):
        super(Client, self).__init__()
        self.id = id
        self.update_step = update_step ## task-level inner update steps
        self.update_step_test = update_step_test
        self.net = deepcopy(model)
        self.base_lr = base_lr
        self.meta_lr = meta_lr
        self.batch_size = batch_size
        self.data = torch.load(path)
        np.random.shuffle(self.data)
    
        self.mode = mode
        self.time = 0
        self.epoch = 0
        if self.mode == "fed_train":
            support_size = int(len(self.data)*1.0)
            support_set = DatasetSplit(self.data[:support_size])
            self.support_loader = DataLoader(
                support_set, batch_size = self.batch_size, shuffle=True,drop_last=True)
            
        elif self.mode == "reptile_train":
            support_size = int(len(self.data)*1.0)
            support_set = DatasetSplit(self.data[:support_size])
            self.support_loader = DataLoader(
                support_set, batch_size = int(0.2*len(support_set)), shuffle=True,drop_last=True)
        
        else:
            support_size = int(len(self.data)*0.5)
            support_set = DatasetSplit(self.data[:support_size])
            self.support_loader = DataLoader(
                support_set, batch_size = self.batch_size, shuffle=True,drop_last=True)          

        query_size = int(len(self.data)*1.0)


        if self.mode == "fed_train":
            pass
        elif self.mode == "reptile_train":
            pass
        else:
            query_set = DatasetSplit(self.data[support_size:query_size])
            self.query_loader = DataLoader(
                query_set, batch_size = self.batch_size, shuffle=True,drop_last=True)
          
        self.optim = torch.optim.Adam(self.net.parameters(), lr = self.base_lr)
        self.outer_optim = torch.optim.Adam(self.net.parameters(), lr = self.meta_lr)
        # self.batch_size = batch_size
        self.size = query_size
        self.device = device
        self.loss_function = torch.nn.CrossEntropyLoss().to(self.device)
        
    def forward(self):
        pass

    def local_fed_train(self):
        for _ in range(self.update_step):          
            i = 0
            for support in self.support_loader:
                support_x, support_y = support
                
                if torch.cuda.is_available():
                    support_x = support_x.cuda(self.device)
                    support_y = support_y.cuda(self.device)
                self.optim.zero_grad()          

                output = self.net(support_x)
                #output = torch.squeeze(output)
                loss = self.loss_function(output,support_y) 
            
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()
                i+=1
                if i >5:
                    break

    def local_meta_reptile_train(self):
        for _ in range(self.update_step):
            net_tem = deepcopy(self.net)
            meta_optim_tem = torch.optim.Adam(net_tem.parameters(), lr = self.base_lr)
            i = 1
            for support in self.support_loader:
                support_x, support_y = support
                
                if torch.cuda.is_available():
                    support_x = support_x.cuda(self.device)
                    support_y = support_y.cuda(self.device)
                
                meta_optim_tem.zero_grad()
                output = net_tem(support_x,)
                output = torch.squeeze(output)
                loss = self.loss_function(output,support_y)
                loss.backward()
                meta_optim_tem.step()

                i += 1
                if i > 5:
                    break

            self.outer_optim.zero_grad()
            for w, w_t in zip(self.net.parameters(), net_tem.parameters()):
                if w.grad is None:
                    w.grad = Variable(torch.zeros_like(w)).to(self.device)
                w.grad.data.add_(w.data - w_t.data)

            self.outer_optim.step()
    
    

    def local_asymeta_fomaml_train(self):
        for _ in range(self.update_step):
            #for batch_idx_1,support, batch_idx_2, query in zip(
                    #enumerate(self.support_loader,self.query_loader)):
             
            spt_batch_idx = random.randint(0,len(self.support_loader)-1)
            qry_batch_idx = random.randint(0,len(self.query_loader)-1)
            support_x, support_y = next(itertools.islice(self.support_loader, spt_batch_idx, None))
            query_x, query_y = next(itertools.islice(self.query_loader, qry_batch_idx, None))
            
            if torch.cuda.is_available():
                support_x = support_x.cuda(self.device)
                support_y = support_y.cuda(self.device)
                query_x = query_x.cuda(self.device)
                query_y = query_y.cuda(self.device)
                
                
            self.optim.zero_grad()                
            output = self.net(support_x)
                #output = torch.squeeze(output)
            loss = self.loss_function(output,support_y) 
                
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            self.outer_optim.zero_grad()
            #for batch_idx, query_x in enumerate(query_x):
            output = self.net(query_x)
                    #output = torch.squeeze(output)
            loss = self.loss_function(output, query_y)
                
            loss.backward()
            self.outer_optim.step()
            self.outer_optim.zero_grad()
                    #break
 
        time_range_1, time_range_2 = 3, 20#------------------------------------
        self.time = random.uniform(time_range_1, time_range_2)
    
    def local_asymeta_reptile_train(self,):        
        for _ in range(self.update_step):
            net_tem = deepcopy(self.net)
            meta_optim_tem = torch.optim.Adam(net_tem.parameters(), lr = self.base_lr)
            i = 1
            for support in self.support_loader:
                support_x, support_y = support
                
                if torch.cuda.is_available():
                    support_x = support_x.cuda(self.device)
                    support_y = support_y.cuda(self.device)
                
                meta_optim_tem.zero_grad()
                output = net_tem(support_x,)
                output = torch.squeeze(output)
                loss = self.loss_function(output,support_y)
                loss.backward()
                meta_optim_tem.step()

                i += 1
                if i > 5:
                    break

            self.outer_optim.zero_grad()
            for w, w_t in zip(self.net.parameters(), net_tem.parameters()):
                if w.grad is None:
                    w.grad = Variable(torch.zeros_like(w)).to(self.device)
                w.grad.data.add_(w.data - w_t.data)

            self.outer_optim.step()
        time_range_1, time_range_2 = 5, 35
        self.time = random.uniform(time_range_1, time_range_2)

    def refresh(self,model):
        for w,w_t in zip(self.net.parameters(),model.parameters()):
            w.data.copy_(w_t.data)
        
    def test(self):
        fine_tuning_loss_list = torch.zeros([self.update_step_test])
        test_loss_list = torch.zeros([self.update_step_test])
        init_loss_list = torch.zeros([1])
        #---
        init_acc_list = torch.zeros([1])
        fine_tuning_acc_list = torch.zeros([self.update_step_test])
        test_acc_list = torch.zeros([self.update_step_test])
        #---
        init_pre_list = torch.zeros([1])
        fine_tuning_pre_list = torch.zeros([self.update_step_test])
        test_pre_list = torch.zeros([self.update_step_test])
        #---
        init_recall_list = torch.zeros([1])
        fine_tuning_recall_list = torch.zeros([self.update_step_test])
        test_recall_list = torch.zeros([self.update_step_test])
        #---
        init_f1_list = torch.zeros([1])
        fine_tuning_f1_list = torch.zeros([self.update_step_test])
        test_f1_list = torch.zeros([self.update_step_test])
       
        
        loss_all, correct_all, total = 0.0, 0.0, 0.0
        precision_all, recall_all, f1_all = 0.0, 0.0, 0.0
        for query in self.query_loader:#---------------------------------------init
            query_x, query_y = query
            if torch.cuda.is_available():
                query_x = query_x.cuda(self.device)
                query_y = query_y.cuda(self.device)
            output = self.net(query_x)
            output = torch.squeeze(output)
            loss = self.loss_function(output, query_y)
            loss_all += loss.item()
            total += len(query_y)
            
            y_hat = self.net(query_x)
            query_pred = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
            correct = torch.eq(query_pred, query_y).sum().item()
            correct_all += correct
            #acc = correct/len(query_x)
            #------------------------------------------------------------------
            #acc_sk = accuracy_score(query_pred, query_y)
            query_y_c = query_y.cpu().numpy()#----
            query_pred_c = query_pred.cpu().numpy()#----
            precision = precision_score(query_pred_c, query_y_c,average='macro')
            recall = recall_score(query_pred_c, query_y_c,average='macro') 
            f1 = f1_score(query_pred_c, query_y_c,average='macro') 
            precision_all += precision
            recall_all += recall
            f1_all += f1
            
        init_loss_list = loss_all/len(self.query_loader)
        init_acc_list = correct_all/total
        init_pre_list = precision_all/len(self.query_loader)
        init_recall_list = recall_all/len(self.query_loader)
        init_f1_list = f1_all/len(self.query_loader)
    
        
        for epoch in range(self.update_step_test):          
            loss_all, correct_all, total = 0.0, 0.0, 0.0
            precision_all, recall_all, f1_all = 0.0, 0.0, 0.0
            for support in self.support_loader:
                self.optim.zero_grad()
                support_x, support_y = support
                if torch.cuda.is_available():
                    support_x = support_x.cuda(self.device)
                    support_y = support_y.cuda(self.device)
                output = self.net(support_x)
                #print(output)
                #output = torch.squeeze(output)
                loss = self.loss_function(output, support_y)
                loss.backward()
                self.optim.step()
                
                y_hat = self.net(support_x)
                query_pred = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
                correct = torch.eq(query_pred, support_y).sum().item()
                correct_all += correct
                #acc = correct/len(support_x)
                support_y_c = support_y.cpu().numpy()#----
                query_pred_c = query_pred.cpu().numpy()#----
                precision = precision_score(query_pred_c, support_y_c,average='macro')
                recall = recall_score(query_pred_c, support_y_c,average='macro') 
                f1 = f1_score(query_pred_c, support_y_c,average='macro')
                
                loss_all += loss.item()
                total += len(support_y)
                precision_all += precision
                recall_all += recall
                f1_all += f1
                
            fine_tuning_loss_list[epoch] = loss_all/len(self.support_loader)
            fine_tuning_acc_list[epoch] = correct_all/total
            fine_tuning_pre_list[epoch] = precision_all/len(self.support_loader)
            fine_tuning_recall_list[epoch] = recall_all/len(self.support_loader)
            fine_tuning_f1_list[epoch] = f1_all/len(self.support_loader)


            
            loss_all, correct_all, total = 0.0, 0.0, 0.0
            precision_all, recall_all, f1_all = 0.0, 0.0, 0.0
            for query in self.query_loader:
                query_x, query_y = query
                if torch.cuda.is_available():
                    query_x = query_x.cuda(self.device)
                    query_y = query_y.cuda(self.device)
                output = self.net(query_x)
                output = torch.squeeze(output)
                loss = self.loss_function(output, query_y)
                loss_all += loss.item()
                
                y_hat = self.net(query_x)
                query_pred = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
                correct = torch.eq(query_pred, query_y).sum().item()
                correct_all += correct
                total += len(query_y)
                query_y_c = query_y.cpu().numpy()#----
                query_pred_c = query_pred.cpu().numpy()#----
                precision = precision_score(query_pred_c, query_y_c,average='macro')
                recall = recall_score(query_pred_c, query_y_c,average='macro') 
                f1 = f1_score(query_pred_c, query_y_c,average='macro')
                precision_all  += precision
                recall_all += recall
                f1_all += f1
        
            test_loss_list[epoch] = loss_all/len(self.query_loader)
            test_acc_list[epoch] = correct_all/total
            test_pre_list[epoch] = precision_all/len(self.query_loader)
            test_recall_list[epoch] = recall_all/len(self.query_loader)
            test_f1_list[epoch] = f1_all/len(self.query_loader)

        fine_tuning_loss_list = fine_tuning_loss_list.detach().numpy()
        test_loss_list = test_loss_list.detach().numpy()
        fine_tuning_acc_list = fine_tuning_acc_list.detach().numpy()
        test_acc_list = test_acc_list.detach().numpy()
        fine_tuning_pre_list = fine_tuning_pre_list.detach().numpy()
        test_pre_list = test_pre_list.detach().numpy()
        fine_tuning_recall_list = fine_tuning_recall_list.detach().numpy()
        test_recall_list = test_recall_list.detach().numpy()
        fine_tuning_f1_list = fine_tuning_f1_list.detach().numpy()
        test_f1_list = test_f1_list.detach().numpy()
        
        return fine_tuning_loss_list,test_loss_list,init_loss_list\
            ,fine_tuning_acc_list,test_acc_list,init_acc_list\
            ,fine_tuning_pre_list,test_pre_list,init_pre_list\
            ,fine_tuning_recall_list,test_recall_list,init_recall_list\
            ,fine_tuning_f1_list,test_f1_list,init_f1_list

      

        
    

        


    
    
 