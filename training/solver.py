from ntpath import join
import os
import numpy as np
import time
from scipy.fft import ifftn
import torch
from torch import optim
from data_loader1 import get_loader
torch.backends.cudnn.deterministic = True
from tqdm import tqdm
import torch.distributed as dist
import gc
from itertools import chain
from fastIntra_utils import selection_sort_k1,selection_sort_k2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from network import *

from matplotlib import pyplot as plt

output=[0,0,0,0,0,0]

class Solver(object):
    def __init__(self, config, gpus):
        self.device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() else 'cpu')

        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Path
        self.model_path = config.model_path
        self.gpus = gpus
        self.build_model()
        #self.load_model()

    def load_model(self):
        def my_load_state_dict(module,state_dict):
            from collections import OrderedDict
            if 'module.' in list(state_dict.keys())[0]:  # multi-gpu training
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove module
                    new_state_dict[name] = v
                module.load_state_dict(new_state_dict, strict = True)
            else:
                module.load_state_dict(state_dict, strict=False)#, strict = True)

        for i,module in enumerate(self.cuSize_list):
            my_load_state_dict(module,torch.load(os.path.join(self.model_path, 'module-%d.pkl' % (i))))
            
    def build_model(self):
        self.res = res1(1)
        
        self.subnet = []
        self.subnet.append(subnet2(6))
        self.subnet.append(subnet3(6))
        self.subnet.append(subnet3(6,x=4,y=8,atten_input=8))
        self.subnet.append(subnet4(6,x=4,y=16,atten_input=8))
        self.subnet.append(subnet4(6,x=4,y=8,atten_input=12))

        self.subnet.append(subnet3(6,x=8,y=4,atten_input=8))
        self.subnet.append(subnet4(6,x=16,y=4,atten_input=8))
        self.subnet.append(subnet4(6,x=8,y=4,atten_input=12))
        
        self.cuSize_list=[self.res,self.subnet[0],self.subnet[1],self.subnet[2],self.subnet[3],self.subnet[4], \
                        self.subnet[5],self.subnet[6],self.subnet[7]]
        self.optimizer=optim.Adam(chain(self.res.parameters(),self.subnet[0].parameters(),self.subnet[1].parameters(), \
                self.subnet[2].parameters(),self.subnet[3].parameters(),self.subnet[4].parameters(), \
                self.subnet[5].parameters(),self.subnet[6].parameters(),self.subnet[7].parameters()) \
                , self.lr, [self.beta1, self.beta2])   
        
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.6, patience=4)

        self.res=self.res.to(self.device)
        for cuSize_idx in range(8):
            self.subnet[cuSize_idx]=self.subnet[cuSize_idx].to(self.device)
        
    def save_model(self):
        for i,module in enumerate(self.cuSize_list):
            tmp_model_path = os.path.join(self.model_path, 'module-%d.pkl' % (i))
            tmp_model = module.state_dict()
            torch.save(tmp_model, tmp_model_path)
    def calculate_loss(self,images,pre,total_acc,total_k2_acc, total_remains,thres=0.1):
        images['gt']=images['gt'].to(self.device)
        sum_loss=0

        soft=torch.nn.functional.softmax(pre[:,0:6],dim=1)
        tmp_sum=torch.sum(torch.log(soft)*images['gt'])
        if torch.isnan(tmp_sum):
            sum_loss-=torch.sum(torch.log(soft+1e-14)*images['gt'])
        else:
            sum_loss-=tmp_sum

        for j in range(images['image'].size(0)):
            if torch.argmax(soft[j])==torch.argmax(images['gt'][j]):
                total_acc+=1
            
            for k in range(6):
                if soft[j][k]>thres:
                    total_k2_acc+=images['gt'][j][k]
                    total_remains+=1

        sum_loss/=images['image'].size(0)
        return total_acc, total_k2_acc, total_remains, sum_loss

    def run(self,cuSize_idx,images,total_acc,total_k2_acc,total_length,total_remains,thres=0.1):

        if cuSize_idx > 4:
            pre = images['image'].transpose(3,2)
        else:
            pre = images['image']

        pre=self.res(pre.to(self.device))
        
        pre=self.subnet[cuSize_idx](pre,images['qp'])

        total_acc, total_k2_acc, total_remains,sum_loss=self.calculate_loss(images,pre,total_acc,total_k2_acc, total_remains,thres)
        
        total_length+=self.batch_size
        return total_acc,total_k2_acc,total_length,total_remains,sum_loss

    def validate(self,valid_loader,cuSize_idx,thres=0.1):
        start = time.time()
        with torch.no_grad():
            for module in self.cuSize_list:
                module.train(False)
                module.eval()
            total_acc=0.
            total_length=0.
            total_remains=0.
            epoch_sum_loss = 0.
            total_k2_acc=0.
            for i, images in enumerate(valid_loader):
                total_acc,total_k2_acc,total_length,total_remains,sum_loss= \
                self.run(cuSize_idx,images,total_acc,total_k2_acc,total_length,total_remains,thres)
                epoch_sum_loss += sum_loss.item()

            print(
                '[Validation]cuSize:%d, Sum_Loss: %.4f, acc: %.4f, k2_acc: %.4f, remains: %.4f, length: %d\n' % \
                    (cuSize_idx, epoch_sum_loss/total_length*self.batch_size,total_acc/total_length, total_k2_acc/total_length, total_remains/total_length,total_length))
            for module in self.cuSize_list:
                module.train(True)
        return epoch_sum_loss/total_length*self.batch_size

    def train(self):
        print("start training")
        train_loader = []
        valid_loader = []
        for cuSize_idx in range(5):
            train_loader.append(get_loader(cuSize=cuSize_idx, batch_size=self.batch_size, num_workers=2, mode='train'))
            valid_loader.append(get_loader(cuSize=cuSize_idx, batch_size=self.batch_size, num_workers=2, mode='valid'))
        sum_image=600000
        validation_num = 10000
        lr = self.lr 
        total_remains=[0]*8
        total_acc=[0.]*8
        total_length=[0.]*8
        epoch_sum_loss=[0.]*8
        total_k2_acc=[0.]*8
        start = time.time()
        while 1:
            for module in self.cuSize_list:
                module.train(True)

            for i, images in enumerate(zip(train_loader[0],train_loader[1],train_loader[2],train_loader[3],train_loader[4])):

                for cuSize_idx in range(8):

                    image_idx = cuSize_idx
                    if cuSize_idx>4:
                        image_idx = cuSize_idx-3

                    total_acc[cuSize_idx], total_k2_acc[cuSize_idx], total_length[cuSize_idx],total_remains[cuSize_idx] \
                    ,sum_loss = \
                    self.run(cuSize_idx, images[image_idx],total_acc[cuSize_idx],total_k2_acc[cuSize_idx], \
                    total_length[cuSize_idx],total_remains[cuSize_idx])

                    sum_loss.backward()
                    epoch_sum_loss[cuSize_idx] += sum_loss.item()

                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if sum_image % 20==0:
                    torch.cuda.empty_cache()
                    gc.collect()

                sum_image-=1
                if sum_image % validation_num==0:
                    print('Training epoch {} times is {}: '.format(sum_image, time.time() - start))
                    valid_loss_all = []
                    for cuSize_idx in range(5):
                        print('[Training]cuSize:%d,Image [%d],Sum_Loss: %.4f, acc: %.4f, k2_acc: %.4f, remains: %.4f,length: %d\n' % \
                            (cuSize_idx, sum_image, epoch_sum_loss[cuSize_idx],total_acc[cuSize_idx]/total_length[cuSize_idx], \
                            total_k2_acc[cuSize_idx]/total_length[cuSize_idx], total_remains[cuSize_idx]/total_length[cuSize_idx],total_length[cuSize_idx]))
                
                        #validation
                        valid_loss_all.append(self.validate(valid_loader[cuSize_idx],cuSize_idx))

                    self.scheduler.step(sum(valid_loss_all))
                    print(self.optimizer.param_groups[0]['lr'])
                    self.save_model()
                    total_remains=[0]*8
                    total_acc=[0.]*8
                    total_length=[0.]*8
                    epoch_sum_loss=[0.]*8
                    total_k2_acc=[0.]*8
                    start = time.time()
                if sum_image<0:
                    break

            if sum_image<0:
                break    
            print("epoch_end")