from collections import defaultdict
from random import choice
import numpy as np
import json
import argparse
import random
import os
import math
import re
import pickle
import sample as smp
import glob
import sparse as sp #version 0.1.0

special_ch=['all_token','null','*','[','(',')',':','-',']','_',',','+']
vocab_cnt_map = defaultdict(int)


parser = argparse.ArgumentParser()
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--sample_save_path', type=str, default='bert_sample/')
parser.add_argument('--sample_input_path', type=str, default='ori_sample/')

def m_to_dense(ms):
    return ms.todense()
    return ms

def load_sample_list(): #加载路径下的所有文件名到list，路径下包含了所有编译配置所有程序样例的数据，每个程序一个文件
    data_list = []
    f_dir = cfg.sample_input_path
    data_list0=glob.glob(f_dir+"*.pkl") #全部数据，不分变异配置
    for compile_i in range(4):
        compile_tag = str(compile_i)
        data=glob.glob(f_dir+"sample_"+compile_tag+"_*.pkl")
        data_list.append(data) #区分编译配置
    return data_list0, data_list
    
def get_rand_index_pair(): #获得随机的一组不同的编译配置
    i = random.randint(0,3)
    j = random.randint(0,3)
    while i==j:
        j = random.randint(0,3)
    return i,j

def get_rand_sample(except_i,max_i,data_list): #随机获取一个程序样例
    index = random.randint(0,max_i-1)
    while index==except_i:
        index = random.randint(0,max_i-1)
    f_name = data_list[index]
    f = open(f_name, 'rb')
    dt = pickle.load(f)
    return dt,index

def get_rand_func(prog): #随机获得该程序样例的一个函数
    index = random.randint(0,len(prog['func_list'])-1)
    func = prog['func_list'][index]
    while len(func.bb_list) < 2:
        index = random.randint(0,len(prog['func_list'])-1)
        func = prog['func_list'][index] 
    return func

def get_rand_bb(func): #随机获得一个BB
    index = random.randint(0,len(func.bb_list)-1)
    bb = func.bb_list[index]
    return bb

def create_anp_data(f,data_list): 
    #bb with adjm[bb1,bb2]=1
    sample, except_i = get_rand_sample(-1,len(data_list),data_list)
    
    #if random.random() > 0.5: #adj bb
    func = get_rand_func(sample)
    i0=random.randint(0,len(func.bb_list)-1)
    j0=random.randint(0,len(func.bb_list)-1)

    cfg_A = m_to_dense(func.cfg_A) #如果两个BB在CFG中相连，则label为1，否则为0
    if cfg_A[i0,j0]>0:
        f.write(func.bb_list[i0].tokens+'\t'+func.bb_list[j0].tokens+'\t'+str(1)+'\n')
    else:
        f.write(func.bb_list[i0].tokens+'\t'+func.bb_list[j0].tokens+'\t'+str(0)+'\n')

def create_big_data(f,data_list):
    sample1, except_i = get_rand_sample(-1,len(data_list),data_list) #随机获取一个程序样例
    sample2, except_i = get_rand_sample(except_i,len(data_list),data_list)

    if random.random() > 0.5: #different program
        bb1 = get_rand_bb(get_rand_func(sample1)) #随机获取一个BB
        bb2 = get_rand_bb(get_rand_func(sample2))
        f.write(bb1.tokens+'\t'+bb2.tokens+'\t'+str(0)+'\n')
    else:
        bb1 = get_rand_bb(get_rand_func(sample1))
        bb2 = get_rand_bb(get_rand_func(sample1))
        f.write(bb1.tokens+'\t'+bb2.tokens+'\t'+str(1)+'\n')

def create_gc_data(f,data_list):
    i,j = get_rand_index_pair()
    if random.random() > 0.5: #different compile
        sample1, except_i = get_rand_sample(-1,len(data_list[i]),data_list[i]) #随机获取一个程序样例
        sample2, except_i = get_rand_sample(-1,len(data_list[j]),data_list[j])

        bb1 = get_rand_bb(get_rand_func(sample1)) #随机获取一个BB
        bb2 = get_rand_bb(get_rand_func(sample2))
        f.write(bb1.tokens+'\t'+bb2.tokens+'\t'+str(0)+'\n')
    else:
        sample1, except_i = get_rand_sample(-1,len(data_list[i]),data_list[i])
        sample2, except_i = get_rand_sample(except_i,len(data_list[i]),data_list[i])
        bb1 = get_rand_bb(get_rand_func(sample1))
        bb2 = get_rand_bb(get_rand_func(sample2))
        f.write(bb1.tokens+'\t'+bb2.tokens+'\t'+str(1)+'\n')

if __name__ == '__main__':
    cfg = parser.parse_args()
    data_list0, data_list = load_sample_list() #加载数据
    sample_amount=2000000 #要生成的训练数据总量

    f_name = cfg.sample_save_path+'anp_corpus'
    f = open(f_name, 'w')
    for i in range(sample_amount):
        create_anp_data(f,data_list0)
    f.close()

    f_name = cfg.sample_save_path+'big_corpus'
    f = open(f_name, 'w')
    for i in range(sample_amount):
        create_big_data(f,data_list0)
    f.close()

    f_name = cfg.sample_save_path+'gc_corpus'
    f = open(f_name, 'w')
    for i in range(sample_amount):
        create_gc_data(f,data_list)
    f.close()



