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
        self.rdo_param = config.rdo_param

        # Path
        self.model_path = config.model_path
        self.gpus = gpus
        self.build_model()
        self.load_model()

    def load_model(self):
        def my_load_state_dict(module,state_dict):
            from collections import OrderedDict
            if 'module.' in list(state_dict.keys())[0]:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]
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
    
    def calculate_loss(self,images,pre,rdtcost,last_thres, new_thres, rdo_gamma):
        images['gt']=images['gt'].to(self.device)

        soft=torch.nn.functional.softmax(pre[:,0:6],dim=1)
        
        for j in range(images['image'].size(0)):

            last_best_rd = 1000000000000000
            for k in range(6):
                if soft[j][k]>last_thres[k] and images['gt'][j][k]< last_best_rd and images['gt'][j][k]>0.00001:
                    last_best_rd = images['gt'][j][k]
            
            new_best_rd = 1000000000000000
            for k in range(6):
                if soft[j][k]>new_thres[k] and images['gt'][j][k]< new_best_rd and images['gt'][j][k]>0.00001:
                    new_best_rd = images['gt'][j][k]
            
            for k in range(6):
                if new_thres[k] > last_thres[k]:
                    if images['gt'][j][k]==0:
                        continue
                    if soft[j][k]>last_thres[k] and soft[j][k]<new_thres[k]:
                        delta_rd = new_best_rd - images['gt'][j][k]
                        if delta_rd < 0:
                            delta_rd = 0
                            
                        rdtcost[k] += (rdo_gamma*delta_rd/100 - self.rdo_param * images['gt'][j][6+k])

                elif new_thres[k] < last_thres[k]:
                    if images['gt'][j][k]==0:
                        continue
                    if soft[j][k]>new_thres[k] and soft[j][k]<last_thres[k]:
                        delta_rd = images['gt'][j][k] - last_best_rd
                        if delta_rd > 0:
                            delta_rd = 0
                        rdtcost[k] += (rdo_gamma*delta_rd/100 + self.rdo_param * images['gt'][j][6+k])
        
        return [rdtcost[k] for k in range(12)]

    def run(self,cuSize_idx,images,rdtcost,last_thres,new_thres):

        if cuSize_idx > 4:
            pre = images['image'].transpose(3,2)
        else:
            pre = images['image']

        pre=self.res(pre.to(self.device))
        
        pre=self.subnet[cuSize_idx](pre,images['qp'])
        
        rdo_gamma = 1.
        if cuSize_idx == 1:
            rdo_gamma = 0.1141
        elif cuSize_idx == 2 or cuSize_idx == 5:
            rdo_gamma = 0.1226
        elif cuSize_idx == 3 or cuSize_idx == 6:
            rdo_gamma = 0.0390
        elif cuSize_idx == 4 or cuSize_idx == 7:
            rdo_gamma = 0.0870

        rdtcost_list = self.calculate_loss(images,pre,rdtcost,last_thres, new_thres, rdo_gamma)
        return rdtcost_list
        

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
            last_thres = [0.125,0.125,0.125,0.125,0.125,0.125]
            new_thres = [0.1,0.1,0.1,0.1,0.1,0.1]
            current_range = [[-0.1,0.31], [-0.1,0.31], [-0.1,0.31], [-0.1,0.31], [-0.1,0.31], [-0.1,0.31]]
            for iter in range(10):
                rdtcost = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
                for i, images in enumerate(valid_loader):
                    rdtcost = self.run(cuSize_idx,images,rdtcost,last_thres,new_thres)
                for k in range(6):
                    if new_thres[k] < last_thres[k] and rdtcost[k] > 0:
                        current_range[k][0]=new_thres[k]
                        if current_range[k][1] == 0.31:
                            new_thres[k] = last_thres[k] + 0.025
                    elif new_thres[k] >= last_thres[k] and rdtcost[k] > 0:
                        current_range[k][1]=new_thres[k]
                    elif new_thres[k] < last_thres[k] and rdtcost[k] < 0:
                        current_range[k][1] = last_thres[k]
                        if current_range[k][0] == -0.1:
                            new_thres[k] = new_thres[k] - 0.025
                            last_thres[k] = last_thres[k] - 0.025
                    elif new_thres[k] >= last_thres[k] and rdtcost[k] < 0:
                        current_range[k][0] = last_thres[k]
                        if current_range[k][1] == 0.31:
                            new_thres[k] = new_thres[k] + 0.025
                            last_thres[k] = last_thres[k] + 0.025
                    
                    if current_range[k][0] != -0.1 and current_range[k][1] != 0.31:    
                        last_thres[k] = (current_range[k][0] + current_range[k][1] * 2) / 3
                        new_thres[k] = (current_range[k][0] * 2 + current_range[k][1]) / 3
                    
                print(cuSize_idx, iter, current_range)
                print(" ")
            print("final", [(current_range[k][0]+current_range[k][1]) / 2 for k in range(6)])

    def train(self):
        print("start training")
        train_loader = []
        valid_loader = []
        for cuSize_idx in range(5):
            valid_loader.append(get_loader(cuSize=cuSize_idx, batch_size=self.batch_size, num_workers=2, mode='valid'))
            
        for cuSize_idx in range(8):
            if cuSize_idx>4:
                image_idx = cuSize_idx-3
            else:
                image_idx = cuSize_idx
            self.validate(valid_loader[image_idx],cuSize_idx)
                