import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
OUTNUM=6

class subset(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(subset,self).__init__()
        self.ch_in=ch_in
        self.ch_out=ch_out
        self.net = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x1 = self.net(x)
        x3=x1+x[...,2:-2,2:-2]
        return x3

class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out,kernel_size=3,stride2=1,padding2=0):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size,stride=stride2,padding=padding2,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x
class subnet2(nn.Module):
    def __init__(self,out_dim):
        super(subnet2,self).__init__()
        self.Conv1=single_conv(ch_in=16,ch_out=16,kernel_size=4,stride2=4,padding2=0)
        self.Conv2=single_conv(ch_in=16,ch_out=32,kernel_size=4,stride2=4,padding2=0)
        self.Conv3=single_conv(ch_in=32,ch_out=128,kernel_size=2,stride2=2,padding2=0)
        self.fc1=nn.Linear(128,64)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(64,out_dim)
        self.atten_module_1=nn.Parameter(torch.ones(4,16))
        self.atten_module_2=nn.Parameter(torch.ones(4,128))
    def forward(self,x,qp_list):
        res=x
        atten_value_1=self.atten_module_1[qp_list]
        res=res*(atten_value_1.view(atten_value_1.shape[0],atten_value_1.shape[1],1,1))
        
        res=self.Conv1(res)
        res=self.Conv2(res)
        res=self.Conv3(res)
        res=res.view(x.shape[0],128)
        res2=res

        atten_value_2=self.atten_module_2[qp_list]
        res2=res2*atten_value_2

        res2=self.fc1(res2)
        res2=self.relu(res2)
        res2=self.fc2(res2)
        #res2=torch.nn.functional.softmax(res2,dim=1) #gen_pth时使用
        return res2
class subnet3(nn.Module):
    def __init__(self,out_dim,x=4,y=4,atten_input=16):
        super(subnet3,self).__init__()
        indim=16
        self.Conv1=single_conv(ch_in=16,ch_out=32,kernel_size=(x,y),stride2=(x,y),padding2=0)
        self.Conv2=single_conv(ch_in=32,ch_out=32,kernel_size=2,stride2=2,padding2=0)
        self.Conv3=single_conv(ch_in=32,ch_out=64,kernel_size=2,stride2=2,padding2=0)
        self.fc1=nn.Linear(64,64)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(64,out_dim)
        self.atten_module_1=nn.Parameter(torch.ones(atten_input,16))
        self.atten_module_2=nn.Parameter(torch.ones(atten_input,64))
    def forward(self,x,qp_list):
        res=x
        atten_value_1=self.atten_module_1[qp_list]
        res=res*(atten_value_1.view(atten_value_1.shape[0],atten_value_1.shape[1],1,1))
        res=self.Conv1(res)
        res=self.Conv2(res)
        res=self.Conv3(res)
        res=res.view(x.shape[0],64)
        res2=res
        atten_value_2=self.atten_module_2[qp_list]
        res2=res2*atten_value_2
        res2=self.fc1(res2)
        res2=self.relu(res2)
        res2=self.fc2(res2)
        #res2=torch.nn.functional.softmax(res2,dim=1)
        return res2
    
class subnet4(nn.Module): #min(h,w)==8
    def __init__(self,out_dim,x=4,y=4,atten_input=16):
        super(subnet4,self).__init__()
        indim=16
        self.Conv1=single_conv(ch_in=16,ch_out=32,kernel_size=(x,y),stride2=(x,y),padding2=0)
        self.Conv2=single_conv(ch_in=32,ch_out=64,kernel_size=2,stride2=2,padding2=0)
        self.fc1=nn.Linear(64,64)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(64,out_dim)
        self.atten_module_1=nn.Parameter(torch.ones(atten_input,16))
        self.atten_module_2=nn.Parameter(torch.ones(atten_input,64))
    def forward(self,x,qp_list):
        res=x
        atten_value_1=self.atten_module_1[qp_list]
        res=res*(atten_value_1.view(atten_value_1.shape[0],atten_value_1.shape[1],1,1))
        res=self.Conv1(res)
        res=self.Conv2(res)
        res=res.view(x.shape[0],64)
        res2=res
        atten_value_2=self.atten_module_2[qp_list]
        res2=res2*atten_value_2
        res2=self.fc1(res2)
        res2=self.relu(res2)
        res2=self.fc2(res2)
        #res2=torch.nn.functional.softmax(res2,dim=1)
        return res2

class res1(nn.Module):
    def __init__(self,ch_in):
        super(res1,self).__init__()
        indim=16
        self.Conv1 = single_conv(ch_in=ch_in,ch_out=indim)
        self.subset1=subset(ch_in=indim,ch_out=indim)
        self.subset2=subset(ch_in=indim,ch_out=indim)
        self.subset3=subset(ch_in=indim,ch_out=indim)
        
    def forward(self,x):
        res=self.Conv1(x)
        res=self.subset1(res)
        res=self.subset2(res)
        res=self.subset3(res)
        return res
