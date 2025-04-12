import torch
import numpy as np
import cv2

# ==========
# YUV420P
# ==========

__all__ = ['import_yuv']

def import_yuv_10bit(seq_path, h, w, tot_frm, yuv_type='420p', start_frm=0, only_y=False): #https://blog.csdn.net/summerlq/article/details/125908008
    if yuv_type == '420p':
        hh, ww = h // 2, w // 2
    elif yuv_type == '444p':
        hh, ww = h, w
    else:
        raise Exception('yuv_type not supported.')

    y_size, u_size, v_size = h * w * 2, hh * ww * 2, hh * ww * 2
    blk_size = y_size + u_size + v_size
    
    # init
    y_seq = np.zeros((tot_frm, h, w), dtype=np.float)
    if not only_y:
        u_seq = np.zeros((tot_frm, hh, ww), dtype=np.float)
        v_seq = np.zeros((tot_frm, hh, ww), dtype=np.float)
    
    with open(seq_path, 'rb') as fp:
        for i in range(tot_frm):
            fp.seek(int(blk_size * (start_frm + i)) * 8, 0)  # skip frames
            y_frm = np.fromfile(fp, dtype=np.uint16, count=y_size//2).reshape(h, w)/4
            if only_y:
                y_seq[i, ...] = y_frm
                '''for m in range(h):
                    for n in range(w):
                        y_seq[i, m, n] = float(ord(fp.read(1))+(ord(fp.read(1))<<8))/4'''
            else:
                '''for m in range(h):
                    for n in range(w):
                        y_seq[i, m, n] = float(ord(fp.read(1))+(ord(fp.read(1))<<8))/4

                for m in range(hh):
                    for n in range(ww):
                        u_seq[i, m, n] = float(ord(fp.read(1))+(ord(fp.read(1))<<8))/4
                for m in range(hh):
                    for n in range(ww):
                        v_seq[i, m, n] = float(ord(fp.read(1))+(ord(fp.read(1))<<8))/4'''
                u_frm = np.fromfile(fp, dtype=np.uint16, \
                    count=u_size//2).reshape(hh, ww)/4
                v_frm = np.fromfile(fp, dtype=np.uint16, \
                    count=v_size//2).reshape(hh, ww)/4
                y_seq[i, ...], u_seq[i, ...], v_seq[i, ...] = y_frm, u_frm, v_frm
    if only_y:
        return y_seq
    else:
        return y_seq, u_seq, v_seq

def import_yuv_4frame(seq_path, h, w, tot_frm, yuv_type='420p', start_frm=0, only_y=False):
    if "MarketPlace" in seq_path or \
        "RitualDance" in seq_path or \
        "DaylightRoad2" in seq_path or \
        "FoodMarket4" in seq_path or \
        "ParkRunning3" in seq_path or \
        "Tango2" in seq_path or \
        "Campfire" in seq_path or \
        "CatRobot" in seq_path:
        return import_yuv_10bit(seq_path, h, w, tot_frm, yuv_type, start_frm, only_y)
    # setup params
    if yuv_type == '420p':
        hh, ww = h // 2, w // 2
    elif yuv_type == '444p':
        hh, ww = h, w
    else:
        raise Exception('yuv_type not supported.')

    y_size, u_size, v_size = h * w, hh * ww, hh * ww
    blk_size = y_size + u_size + v_size
    
    # init
    y_seq = np.zeros((tot_frm, h, w), dtype=np.uint8)
    if not only_y:
        u_seq = np.zeros((tot_frm, hh, ww), dtype=np.uint8)
        v_seq = np.zeros((tot_frm, hh, ww), dtype=np.uint8)

    # read data
    with open(seq_path, 'rb') as fp:
        for i in range(tot_frm):
            fp.seek(int(blk_size * (start_frm + i) * 8), 0)  # skip frames
            y_frm = np.fromfile(fp, dtype=np.uint8, count=y_size).reshape(h, w)
            if only_y:
                y_seq[i, ...] = y_frm
            else:
                u_frm = np.fromfile(fp, dtype=np.uint8, \
                    count=u_size).reshape(hh, ww)
                v_frm = np.fromfile(fp, dtype=np.uint8, \
                    count=v_size).reshape(hh, ww)
                y_seq[i, ...], u_seq[i, ...], v_seq[i, ...] = y_frm, u_frm, v_frm

    if only_y:
        return y_seq
    else:
        return y_seq, u_seq, v_seq


def import_yuv(seq_path, h, w, tot_frm, yuv_type='420p', start_frm=0, only_y=False):
    if "MarketPlace" in seq_path or \
        "RitualDance" in seq_path or \
        "DaylightRoad2" in seq_path or \
        "FoodMarket4" in seq_path or \
        "ParkRunning3" in seq_path or \
        "Tango2" in seq_path or \
        "Campfire" in seq_path or \
        "CatRobot" in seq_path:
        return import_yuv_10bit(seq_path, h, w, tot_frm, yuv_type, start_frm, only_y)
    # setup params
    if yuv_type == '420p':
        hh, ww = h // 2, w // 2
    elif yuv_type == '444p':
        hh, ww = h, w
    else:
        raise Exception('yuv_type not supported.')

    y_size, u_size, v_size = h * w, hh * ww, hh * ww
    blk_size = y_size + u_size + v_size
    
    # init
    y_seq = np.zeros((tot_frm, h, w), dtype=np.uint8)
    if not only_y:
        u_seq = np.zeros((tot_frm, hh, ww), dtype=np.uint8)
        v_seq = np.zeros((tot_frm, hh, ww), dtype=np.uint8)

    # read data
    with open(seq_path, 'rb') as fp:
        for i in range(tot_frm):
            fp.seek(int(blk_size * (start_frm + i)), 0)  # skip frames
            y_frm = np.fromfile(fp, dtype=np.uint8, count=y_size).reshape(h, w)
            if only_y:
                y_seq[i, ...] = y_frm
            else:
                u_frm = np.fromfile(fp, dtype=np.uint8, \
                    count=u_size).reshape(hh, ww)
                v_frm = np.fromfile(fp, dtype=np.uint8, \
                    count=v_size).reshape(hh, ww)
                y_seq[i, ...], u_seq[i, ...], v_seq[i, ...] = y_frm, u_frm, v_frm

    if only_y:
        return y_seq
    else:
        return y_seq, u_seq, v_seq

def selection_sort_k1(soft,gt):  #排序soft,找到第N大的值在gt中:
    index=[0,1,2,3,4,5] #index[k]表示现在排在第k大的数原来的下标
    for i in range(6):
        k=i
        temp = soft[i]

        #k 是i+1到6之间最大的数的下标
        for j in range(i+1,6):
            if soft[j]>temp:
                temp = soft[j]
                k = j
        
        #soft[i]和soft[k]调换位置
        if k!=i:
            temp_index = index[k]
            soft[k] = soft[i]
            index[k]=index[i]
            soft[i] = temp
            index[i]=temp_index
        
        if gt[index[i]]==1:
            return i, gt[index[0]], index[0]
        
def selection_sort_k2(soft,gt,pre_k2):  #排序soft,找到第N大的值使得gt都小于等于第N大的值:
    index=[0,1,2,3,4,5]
    for i in range(6):
        k=i
        temp = soft[i]
        for j in range(i+1,6):
            if soft[j]>temp:
                temp = soft[j]
                k = j
        temp_index = index[k]
        
        #soft[i]和soft[k]调换位置
        soft[k] = soft[i]
        index[k]=index[i]
        soft[i] = temp
        index[i]=temp_index
    k2_acc=0
    for i in range(6-pre_k2):
        k2_acc+=(gt[index[i]]==1)
    for i in range(6):
        if gt[index[5-i]]==1:
            return i, k2_acc
def test_sort(input_soft,pre_k2):  #按照soft排序，取出前k大的
    soft=input_soft.clone()
    ret_list=[0,0,0,0,0,0]
    index=[0,1,2,3,4,5]
    #print(6-int(pre_k2))
    for i in range(min(3,max(1,6-int(pre_k2+0.8)))):
    #for i in range(max(1,6-int(pre_k2+0.5))):
        k=i
        temp = soft[i]
        for j in range(i+1,6):
            if soft[j]>temp:
                temp = soft[j]
                k = j
        temp_index = index[k]
        
        #soft[i]和soft[k]调换位置
        soft[k] = soft[i]
        index[k]=index[i]
        soft[i] = temp
        index[i]=temp_index
        ret_list[temp_index]=1
    return ret_list
def copy_value(a,b):
    c=a
    d=b
    return c,d