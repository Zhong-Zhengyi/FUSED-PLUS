import time
import os
import copy
from unittest import result
import torch
# from torch import tensor
from torch.nn import parameter

import torch.nn.functional as F
import torch.optim as optim 
import numpy as np
from models.AlexNet import AlexNet, AlexNet_UL
import time
import random

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class TrainerPrivate(object):
    def __init__(self, model, device, dp, sigma,num_classes, forget_paradigm, ul_model, args):
        self.model = model
        self.device = device
        self.dp = dp
        self.sigma = sigma
        self.num_classes = num_classes
        self.forget_paradigm = forget_paradigm
        self.ul_mode = ul_model
        self.args = args

    def _local_update(self,dataloader, local_ep, lr, optimizer, ul_mode_train='none'):
        # self.optimizer = optim.SGD(self.model.parameters(),
        #                         lr,
        #                         momentum=0.9,
        #                         weight_decay=0.0005)
        self.optimizer = optimizer
        epoch_loss = []
        train_ldr = dataloader 
        update_local_ep={}
        for param_tensor in self.model.state_dict():
            if "weight" in param_tensor or "bias" in param_tensor:
                update_local_ep[param_tensor] = torch.zeros_like(self.model.state_dict()[param_tensor]).to(self.device)
        for epoch in range(local_ep):
            loss_meter = 0
            acc_meter = 0

            for batch_idx, batch in enumerate(train_ldr):
                if self.args.data_name == 'text':
                    x, at, y = batch[0].to(self.device, dtype=torch.long), batch[1].to(self.device), batch[2].to(self.device)
                else:
                    x, y = batch[0].to(self.device), batch[1].to(self.device)

                self.optimizer.zero_grad()

                loss = torch.tensor(0.).to(self.device)

                if self.args.data_name == 'text':
                    pred = self.model(x, at)
                    pred = pred.logits
                else:
                    pred = self.model(x)
    
                loss += F.cross_entropy(pred, y)
                
                acc_meter += accuracy(pred, y)[0].item()
                loss.backward()

                self.optimizer.step() 
                loss_meter += loss.item()

            loss_meter /= len(train_ldr)
            
            acc_meter /= len(dataloader)

            epoch_loss.append(loss_meter)
                        
        if self.dp:
            for param in self.model.parameters():
                param.data = param.data + torch.normal(torch.zeros(param.size()), self.sigma).to(self.device)
       
        return self.model.state_dict(), np.mean(epoch_loss), update_local_ep
    
    
    def _local_update_ul(self,dataloader, local_ep, lr, optimizer, ul_class_id, ul_mode_train=None):
        if ul_mode_train ==None:
            ul_mode_train=self.ul_mode
        else:
            ul_mode_train=ul_mode_train
        # print('ul_mode_train:',ul_mode_train)

        fine_tune_mode=0
        if fine_tune_mode==1:
            for name,param in self.model.named_parameters():
                if 'ul' not in name:
                    param.requires_grad=False
                else:
                    param.requires_grad = True
                    # print('Fine tune part:',name)

    
        self.optimizer = optimizer
              
        epoch_loss = []
        normalize_loss = []
        classify_loss=[]
        ul_acc=[]
        train_ldr = dataloader 

        for epoch in range(local_ep):      
            loss_meter = 0
            classifier_loss_meter=0
            norm_loss_meter=0
            acc_meter = 0
            num_classes=(self.num_classes)
            
            model_mode='SOV_model'
            
            for batch_idx, batch in enumerate(train_ldr):
                if self.args.data_name == 'text':
                    x, at, y = batch[0].to(self.device, dtype=torch.long), batch[1].to(self.device), batch[2].to(self.device)
                else:
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                
                ground_labels=y
               
                if self.forget_paradigm=='sample' or self.forget_paradigm=='client':   #self.ul_mode=='ul_samples' or self.ul_mode=='ul_samples_backdoor' or 'u_samples_whole_client: # random false labels
                    # print('ul_mode：',self.ul_mode)
                    one_hot_labels=[]
                    for label in y:
                        if label < (self.num_classes):
                            one_hot_label=torch.nn.functional.one_hot(label, self.num_classes).unsqueeze(0).to(self.device, dtype=torch.int64)
                            one_hot_label=torch.cat((one_hot_label,one_hot_label),dim=1)
                            
                        else:
                            true_label= label % (self.num_classes)
                            label_a=torch.nn.functional.one_hot(true_label, self.num_classes).unsqueeze(0).to(self.device, dtype=torch.int64)

                            random_label=(label-label%(self.num_classes))/(self.num_classes)-1
                            random_label=random_label.to(dtype=torch.int64)
                            # print('random_label',true_label)
                            label_b=torch.nn.functional.one_hot(random_label, self.num_classes).unsqueeze(0).to(self.device, dtype=torch.int64)
                            one_hot_label=torch.cat((label_a,label_b),dim=1)
                        
                        one_hot_labels.append(one_hot_label)
                    
                    labels_batch=torch.cat(one_hot_labels,dim=0)
                    labels_batch=labels_batch.to(self.device)
                  
                elif self.forget_paradigm=='class':

                    one_hot_labels=[]
                    for label in y:
                        one_hot_label_ul=torch.nn.functional.one_hot(torch.tensor(ul_class_id), self.num_classes).unsqueeze(0).to(self.device, dtype=torch.int64)
                        one_hot_label_true=torch.nn.functional.one_hot(label, self.num_classes).unsqueeze(0).to(self.device, dtype=torch.int64)
                        one_hot_label=torch.cat((one_hot_label_true,one_hot_label_ul),dim=1)
                        one_hot_labels.append(one_hot_label)
                            
                    labels_batch = torch.cat(one_hot_labels,dim=0)
                    labels_batch = labels_batch.to(self.device)

                self.optimizer.zero_grad()

                loss = torch.tensor(0.).to(self.device)

                if self.args.data_name == 'text':
                    pred = self.model(x, at)
                else:
                    pred = self.model(x)

                if model_mode=='MIA_model':
                    one_hot_loss=one_hot_CrossEntropy()
                    #print(pred.size(),y.size())
                    loss +=one_hot_loss(pred, labels_batch,num_classes)

                    acc_meter += accuracy(pred[:,0:(self.num_classes)], ground_labels)[0].item()
                elif model_mode=='SOV_model':

                    # prob_a=torch.nn.functional.softmax(pred[0:10], dim=1)
                    # pred_b=torch.nn.functional.softmax(pred[10:20], dim=1)
                    prob_a=torch.nn.functional.softmax(pred[:,0:num_classes]) #softmax
                    #print(a.size())
                    prob_b=torch.nn.functional.softmax(pred[:,num_classes:2*num_classes])
                    # print(b.size())
                    prob=torch.cat((prob_a,prob_b),dim=1)

                    one_hot_loss = one_hot_CrossEntropy()
                    utility_loss=one_hot_loss(prob, labels_batch,num_classes)

                    # L2_loss=torch.nn.MSELoss()
                    # max_logits,_=torch.max(pred,dim=1)
                    # norm_loss=0.1*L2_loss(max_logits,torch.full([pred.size(0)],10.0).to(self.device))
                    #print(pred.size(),y.size())
                    
                    # loss +=utility_loss+norm_loss
                    loss += utility_loss

                    # loss += F.cross_entropy(pred, y)
                    acc_meter += accuracy(pred[:,0:num_classes], ground_labels)[0].item()

                loss.backward(retain_graph=True)
                self.optimizer.step() 
                loss_meter += loss.item()
                classifier_loss_meter+=utility_loss.item()
                # norm_loss_meter+=norm_loss.item() 

            loss_meter /= len(train_ldr)
            classifier_loss_meter /=len(train_ldr)
            norm_loss_meter /=len(train_ldr)
            
            acc_meter /= len(dataloader)

            epoch_loss.append(loss_meter)
            normalize_loss.append(norm_loss_meter)
            classify_loss.append(classifier_loss_meter)
                        
        if self.dp:
            for param in self.model.parameters():
                param.data = param.data + torch.normal(torch.zeros(param.size()), self.sigma).to(self.device)
        
        return self.model.state_dict(), np.mean(epoch_loss),np.mean(classify_loss), np.mean(normalize_loss)
    
class one_hot_CrossEntropy(torch.nn.Module):

    def __init__(self):
        super(one_hot_CrossEntropy,self).__init__()
    
    def forward(self, x ,y,num_classes):
        # P_i = torch.nn.functional.softmax(x, dim=1)
        # num_classes=100
        loss_a = y[:,0:num_classes] *torch.log(x[:,0:num_classes] + 0.0000001)
        loss_b = y[:,num_classes:2*num_classes] *torch.log(x[:,num_classes:2*num_classes] + 0.0000001)

        loss_a = -torch.mean(torch.sum(loss_a,dim=1),dim = 0)
        loss_b = -torch.mean(torch.sum(loss_b,dim=1),dim = 0)
        return loss_a+loss_b
