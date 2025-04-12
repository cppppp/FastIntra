import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F

from skimage import transform
from PIL import Image
import json
import cv2
import struct
import glob
from fastIntra_utils import *
import time, random
from sys import getsizeof as getsize
import array

class ImageFolder1(data.Dataset):

    def __init__(self,mode,batch_size,cuSize,debug):
        self.debug=debug  #train or debug
        self.mode=mode
        self.cuSize=cuSize
        self.list=self.getlist()
        self.batch_size=batch_size        
        self.max_pool=torch.nn.MaxPool2d((2,2))
        
    def __getitem__(self, index):
        tmp={}

        w = 768
        h = 512

        tmp_path = str(self.list[index][0])
        tmp_hgt = self.list[index][1]
        tmp_left = self.list[index][2]
        tmp_top = self.list[index][3]
        tmp['qp'] = self.list[index][4]
        tmp['gt'] = self.list[index][5]
        
        if self.cuSize == 0:
            tmp_hgt=32
            tmp_wid=32
        elif self.cuSize == 1:
            tmp_hgt=16
            tmp_wid=16
        elif self.cuSize == 2:
            if tmp_hgt==1:
                tmp_hgt=16
                tmp_wid=32
            else:
                tmp_hgt=32
                tmp_wid=16
        elif self.cuSize == 3:
            if tmp_hgt==1:
                tmp_hgt=8
                tmp_wid=32
            else:
                tmp_hgt=32
                tmp_wid=8
        elif self.cuSize == 4:
            if tmp_hgt==1:
                tmp_hgt=8
                tmp_wid=16
            else:
                tmp_hgt=16
                tmp_wid=8

        border = 7

        y = import_yuv("/home/user/year23research/QTMTdataset/raw/"+tmp_path+"_768x512_1.yuv", h, w, 1, yuv_type='420p', start_frm=0, only_y=True)[0] # (h, w)
        if tmp_top-border<0 or tmp_top+tmp_hgt+border>h or tmp_left-border<0 or tmp_left+tmp_wid+border>w:
            #tmp['image']=torch.unsqueeze( \
            #torch.from_numpy(y[border-border:border+border+tmp_hgt, \
            #                border-border:border+border+tmp_wid].copy()), dim=0)
            #pad with same
            padded = np.pad(y, pad_width = border, mode = 'edge').copy()
            padded = torch.from_numpy(padded)
            tmp['image']=torch.unsqueeze( \
            padded[tmp_top:tmp_top+2*border+tmp_hgt, tmp_left:tmp_left+2*border+tmp_wid], dim=0).float()
        else:
            tmp['image']=torch.unsqueeze( \
            torch.from_numpy(y[tmp_top-border:tmp_top+border+tmp_hgt, \
                            tmp_left-border:tmp_left+border+tmp_wid].copy()), dim=0).float()

        def rotate_values():
            new_gt=tmp['gt'].clone()
            new_gt[2], new_gt[3], new_gt[4], new_gt[5]=tmp['gt'][3], tmp['gt'][2], tmp['gt'][5], tmp['gt'][4]
            new_gt[8], new_gt[9], new_gt[10], new_gt[11]=tmp['gt'][9], tmp['gt'][8], tmp['gt'][11], tmp['gt'][10]
            tmp['gt']=new_gt

        if tmp_hgt>tmp_wid:
            tmp['image']=tmp['image'].transpose(2,1) #tmp['image']=tmp['image'].transpose(1,0)
            rotate_values()
            swap = tmp_hgt
            tmp_hgt=tmp_wid
            tmp_wid=swap

        tmp['qp'] = torch.tensor([tmp['qp']])
        
        return tmp

    def getlist(self):
        datalist=[]
        print("getting list")
        def gen_datalist(qp,train_or_test):
            self.yuv_path_list=[]
            name_list=glob.glob("../collected_"+str(self.cuSize)+'/'+str(qp)+"/*")
            self.yuv_list = sorted(name_list, key=lambda x:int(os.path.basename(x).split('_')[0]), reverse=False)
            random.seed(65345)

            self.yuv_path_list = np.array(self.yuv_list)

            if self.cuSize==2:
                portion=22/100
            elif self.cuSize==1 or self.cuSize==3:
                portion=14/100
            elif self.cuSize==4:
                portion=6/100
            else:
                portion=1

            for i,path in enumerate(self.yuv_path_list):
                with open(path,"r") as write_file:
                    cu_pic=json.load(write_file)

                for key,splits in cu_pic.items():
                    if random.random()>portion * 0.5:
                        continue
                    data_item=[]
                    tmp_qp=(qp-22)//5
                    mode = splits[0]

                    if self.cuSize==1:
                        tmp_qp=mode+tmp_qp*4
                    elif self.cuSize==2 or self.cuSize==3:
                        tmp_qp=mode%2+tmp_qp*2
                    elif self.cuSize==4:
                        tmp_qp=mode%3+tmp_qp*3

                    tmp_hgt = 0
                    if self.cuSize==2:
                        if mode//2==0:
                            tmp_hgt=1 #hgt小的记为True
                        else:
                            tmp_hgt=0
                    elif self.cuSize==3:
                        if mode//2==0:
                            tmp_hgt=1
                        else:
                            tmp_hgt=0
                    elif self.cuSize==4:
                        if mode//3==0:
                            tmp_hgt=1
                        else:
                            tmp_hgt=0

                    tmp_top=int(key.split("_")[0])
                    tmp_left=int(key.split("_")[1])
                    tmp_path = int(path.split("/")[-1].split('.')[0].split("_")[0])

                    data_item.append(tmp_path)
                    data_item.append(tmp_hgt)
                    data_item.append(tmp_left)
                    data_item.append(tmp_top)
                    data_item.append(tmp_qp)

                    gt=torch.zeros((12))
                    for z in range(12):
                        gt[z] = splits[z+1]

                    data_item.append(gt)
                    datalist.append(data_item)
                            
        if self.mode=='train':
            print("error!!!!!!!!!!!!!!")
        else:
            gen_datalist(37,'test')
            gen_datalist(32,'test')
            gen_datalist(27,'test')
            gen_datalist(22,'test')

        print("getting list finished")
        print(len(datalist),self.cuSize,self.mode)
        return datalist
    def __len__(self):
        return len(self.list)-len(self.list)%self.batch_size

def collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        collated[key] = torch.stack([sample[key] for sample in batch])
    #print("shape0",collated['qp'].shape)
    collated['qp'] = torch.squeeze(collated['qp'],dim=1)
    #print("shape2",collated['qp'].shape)
    return collated
    
def get_loader(cuSize,batch_size, num_workers=2, mode='train',debug='train'):
    """Builds and returns Dataloader."""
    dataset = ImageFolder1(mode=mode,batch_size=batch_size,cuSize=cuSize,debug=debug)
    if debug=='train':
        return data.DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    collate_fn = collate_fn)
    else:
        return data.DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    collate_fn = collate_fn)
