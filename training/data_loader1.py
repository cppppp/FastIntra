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

        y = import_yuv("/home/user/year23research/saved_from_server/run-10.23/yuv/0/"+tmp_path+"_768x512_1.yuv", h, w, 1, yuv_type='420p', start_frm=0, only_y=True)[0] # (h, w)
        if tmp_top-border<0 or tmp_top+tmp_hgt+border>h or tmp_left-border<0 or tmp_left+tmp_wid+border>w:
            #tmp['image']=torch.unsqueeze( \
            #torch.from_numpy(y[border-border:border+border+tmp_hgt, \
            #                border-border:border+border+tmp_wid].copy()), dim=0)
            #pad with same
            padded = np.pad(y, pad_width = border, mode = 'edge').copy()
            padded = torch.from_numpy(padded)
            tmp['image']=torch.unsqueeze( \
            padded[tmp_top:tmp_top+2*border+tmp_hgt, tmp_left:tmp_left+2*border+tmp_wid], dim=0)
        else:
            tmp['image']=torch.unsqueeze( \
            torch.from_numpy(y[tmp_top-border:tmp_top+border+tmp_hgt, \
                            tmp_left-border:tmp_left+border+tmp_wid].copy()), dim=0)

        def rotate_values():
            new_gt=tmp['gt'].clone()
            new_gt[2], new_gt[3], new_gt[4], new_gt[5]=tmp['gt'][3], tmp['gt'][2], tmp['gt'][5], tmp['gt'][4]
            tmp['gt']=new_gt

        if tmp_hgt>tmp_wid:
            tmp['image']=tmp['image'].transpose(2,1) #tmp['image']=tmp['image'].transpose(1,0)
            rotate_values()
            swap = tmp_hgt
            tmp_hgt=tmp_wid
            tmp_wid=swap
        
        flipped=tmp['image'].clone()

        rand=random.random()
        if self.mode=='train':
            if (rand<0.125 and self.cuSize<2) or (rand<0.25 and self.cuSize>=2): 
                for i in range(tmp_hgt+border*2):
                    flipped[:,tmp_hgt+border*2-1-i]=tmp['image'][:,i]
            elif (rand<0.25 and self.cuSize<2) or (rand<0.5 and self.cuSize>=2):
                for i in range(tmp_wid+border*2):
                    flipped[:,:,tmp_wid+border*2-1-i]=tmp['image'][:,:,i]
            elif (rand<0.375 and self.cuSize<2) or (rand<0.75 and self.cuSize>=2):
                for i in range(tmp_wid+border*2):
                    flipped[:,:,tmp_wid+border*2-1-i]=tmp['image'][:,:,i]
                for i in range(tmp_hgt+border*2):
                    tmp['image'][:,tmp_hgt+border*2-1-i]=flipped[:,i]
                tmp['image']=tmp['image'].float()
                tmp['qp'] = torch.tensor([tmp['qp']])
                return tmp
            elif (rand>0.5 and self.cuSize<=1):
                rotated=tmp['image'].clone()
                rotate_values()
                if self.cuSize==1:
                    if tmp['qp']%4==3 or tmp['qp']%4==2:
                        tmp['qp']=tmp['qp']-tmp['qp']%4+(5-tmp['qp']%4)

                for i in range(tmp_hgt+border*2):
                    rotated[:,i]=tmp['image'][:,:,i]
                
                if rand<0.625:
                    for i in range(tmp_hgt+border*2):
                        flipped[:,tmp_hgt+border*2-1-i]=rotated[:,i]
                elif rand<0.75:
                    for i in range(tmp_wid+border*2):
                        flipped[:,:,tmp_wid+border*2-1-i]=rotated[:,:,i]
                elif rand<0.875:
                    for i in range(tmp_wid+border*2):
                        flipped[:,:,tmp_wid+border*2-1-i]=rotated[:,:,i]
                    for i in range(tmp_hgt+border*2):
                        tmp['image'][:,tmp_hgt+border*2-1-i]=flipped[:,i]
                    tmp['image']=tmp['image'].float()
                    tmp['qp'] = torch.tensor([tmp['qp']])
                    return tmp
                else:
                    flipped=rotated

        tmp['image']=flipped.float()
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

            if self.cuSize==2:
                portion=18/100
                train_list_index = random.sample([i for i in range(len(self.yuv_list))], len(self.yuv_list) *69//70) #24//100
                val_list_index = list(set([i for i in range(len(self.yuv_list))]) - set(train_list_index)) #//200
            elif self.cuSize==1 or self.cuSize==3:
                portion=8/100
                train_list_index = random.sample([i for i in range(len(self.yuv_list))], len(self.yuv_list) *69//70)
                val_list_index = list(set([i for i in range(len(self.yuv_list))]) - set(train_list_index)) #//500
            elif self.cuSize==4:
                portion=5/100
                train_list_index = random.sample([i for i in range(len(self.yuv_list))], len(self.yuv_list) *69//70)
                val_list_index = list(set([i for i in range(len(self.yuv_list))]) - set(train_list_index))
            else: #cuSize==0 or 5
                portion=1
                train_list_index = random.sample([i for i in range(len(self.yuv_list))], len(self.yuv_list) *49//50)
                val_list_index = list(set([i for i in range(len(self.yuv_list))]) - set(train_list_index))
                #print(val_list_index)
            print(self.cuSize,qp,val_list_index)
            
            if train_or_test=='train':
                self.yuv_path_list = np.array(self.yuv_list)[train_list_index]
            else:
                self.yuv_path_list = np.array(self.yuv_list)[val_list_index]

            for i,path in enumerate(self.yuv_path_list):
                '''if i==50:
                    break'''
                with open(path,"r") as write_file:
                    cu_pic=json.load(write_file)

                if self.cuSize%5==0:
                    mode_num=1
                elif self.cuSize>=1 and self.cuSize<=3:
                    mode_num=4
                elif self.cuSize==4:
                    mode_num=6

                for mode in range(mode_num):
                    for key,splits in cu_pic['prob'][mode].items():
                        if random.random()>portion:
                            continue
                        data_item=[]
                        tmp_qp=(qp-22)//5

                        if self.cuSize==1:
                            tmp_qp=mode+tmp_qp*4
                        elif self.cuSize==2 or self.cuSize==3:
                            tmp_qp=mode%2+tmp_qp*2
                        elif self.cuSize==4:
                            tmp_qp=mode%3+tmp_qp*3

                        tmp_hgt = 0
                        if self.cuSize==2:
                            if mode//2==0:
                                tmp_hgt=1
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

                        gt=torch.zeros((6))
                        sum_prob=0
                        for idx,par_mode in enumerate(splits):
                            gt[idx]=par_mode    
                            sum_prob+=par_mode
                        
                        gt/=sum_prob
                        data_item.append(gt)
                        datalist.append(data_item)
                            
        if self.mode=='train':
            start=time.time()
            gen_datalist(37,'train')
            gen_datalist(32,'train')
            gen_datalist(27,'train')
            gen_datalist(22,'train')
            print('Loading dataset times is {}: '.format(time.time() - start))
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
    collated['qp'] = torch.squeeze(collated['qp'],dim=1)
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
