from collections import defaultdict
from random import choice
import pandas as pd
import numpy as np
import json
import argparse
import torch
import torch.nn as nn
import random
from torch.autograd import Variable
import os
import math
import re
import pickle
from sklearn.decomposition import PCA

class Vocab():
    def __init__(self):
        self.token_set = set()
        self.map = defaultdict(str)
        self.low_map = defaultdict(str)
        self.cnt_map = defaultdict(int)
        self.size = 0
        self.token_eb_dim = 1024 #512
        self.low_token_eb_dim = 128
        self.special_ch=['all_token','null','*','[','(',')',':','-',']','_',',','+']
        for token in self.special_ch:
            self.token_set.add(token)

        # for token in self.special_ch:
        #     self.map[token] = np.zeros(self.token_eb_dim)
        #     self.map[token][self.size]=1
        #     self.size += 1
    # def get_eb(self,token):
    #     token = re.sub(r"0x[A-Za-z0-9]{6}","0xaddr",token,0)
    #     token = re.sub(r"0x[A-Za-z0-9]+","0xnum",token,0)
    #     if not (token in self.map.keys()):
    #         print("[Error] map has no key:"+token)
    #         return np.zeros(self.token_eb_dim)
    #     else:
    #         return self.map[token]

    def get_sub_token_for_list(self,token_list): #将读取的tokens分割为多个token的list，如指令 "cmp r8，4" 读取为tokens时为"cmp" "r8," "4" 需要变为 "cmp" "r8" "," "4"
        result=[]
        for token_s in token_list:
            result = result+self.get_sub_token(token_s)
        return result

    def get_sub_token(self,token_s):
        t=''
        sub_token_list=[]
        for ch in token_s:
            if ch not in self.special_ch:
                t = t+ch
            else:
                if len(t)>0:
                    sub_token_list.append(t)
                sub_token_list.append(ch)
                t=''
        if len(t)>0:
            sub_token_list.append(t)
        return sub_token_list

    def add_token(self,tokens):
        token_list = tokens
        for token in token_list:
            token = re.sub(r"0x[A-Za-z0-9]{6}","0xaddr",token,0) #替换地址为0xaddr
            token = re.sub(r"0x[A-Za-z0-9]+","0xnum",token,0)

            if token not in self.token_set: #第一次出现
                self.token_set.add(token)
                self.cnt_map[token] = 0
            else: #已出现过
                self.cnt_map[token]  += 1

            # if token not in self.map.keys():
            #     self.map[token] = np.zeros(self.token_eb_dim)
            #     if self.size<self.token_eb_dim:
            #         self.map[token][self.size] = 1
            #         self.size += 1
            #     else:
            #         print("[Error] vocab size overflow(size,token):",self.size,token)

    def create_eb(self): #废弃，未使用
        pca = PCA(n_components=self.low_token_eb_dim)
        vocab_keys=list(self.token_set)
        if len(vocab_keys)>self.token_eb_dim:
            print("[Error] vocab_keys too large:",len(vocab_keys),self.token_eb_dim)
        matrix=np.zeros((len(vocab_keys),self.token_eb_dim))
        for i in range(len(vocab_keys)):
            matrix[i][i] = 1
            self.map[vocab_keys[i]]=matrix[i]
        
        matrix = pca.fit_transform(matrix)
        for i in range(len(vocab_keys)):
            self.low_map[vocab_keys[i]]=matrix[i]

    def print_vocab(self):
        print("Vocab:",len(self.map.keys()))
        print(self.map.keys())

    def save_dic(self):
        f_name = "vocab_all_type_debug"
        f = open(f_name + '.pkl', 'wb')
        dic_t = {'map':self.map, 'low_map':self.low_map}
        pickle.dump(dic_t, f, pickle.HIGHEST_PROTOCOL)

    def save_cnt_dic(self):
        f_name = "vocab_cnt_all_type"
        f = open(f_name + '.pkl', 'wb')
        pickle.dump(self.cnt_map, f, pickle.HIGHEST_PROTOCOL)

def load_ori(data_path,vocab):    
        if (os.path.getsize(data_path+"_bb")==0) or \
           (os.path.getsize(data_path+"_adj")==0) or \
           (os.path.getsize(data_path+"_arg")==0) or \
           (os.path.getsize(data_path+"_dfg_adj")==0) or \
           (os.path.getsize(data_path+"_dfg_arg")==0) or \
           (os.path.getsize(data_path+"_cg_adj")==0) or \
           (os.path.getsize(data_path+"_cg_arg")==0):
            return
        bb_f = open(data_path+"_bb")
        arg_f = open(data_path+"_arg")
        arg_line = arg_f.readline()
        #TODO： 其实这里不需要读取和遍历filename_arg文件，直接遍历所有的filename_bb文件就可以，构造词汇表时不需要关心这个token来自于哪个程序
        while arg_line: #every func
            func_addr,bb_amount = map(int,arg_line.split())
            arg_line = arg_f.readline()
            bb_ins_amount_list = list(map(int,arg_line.split()))

            for i in range(bb_amount): #every bb
                for j in range(bb_ins_amount_list[i]): #every inst
                    bb_line = bb_f.readline()
                    tmp_s=bb_line.split()
                    try:
                        inst_addr = int(tmp_s[0].rstrip(':'),16) #filename_bb文件中，格式为addr: tokens 如（0x400578: sub rsp, 8）
                    except ValueError:
                        continue
                    del tmp_s[0]  
                    tmp_s = vocab.get_sub_token_for_list(tmp_s)
                    vocab.add_token(tmp_s)  
            
            arg_line = arg_f.readline()
            arg_line = arg_f.readline()

def get_file_in_poj_class(path): #读取这个文件夹下所有程序样例的文件名
    file_id_list=[]
    file_list = os.popen('ls '+path+'*_bb') #因为每个程序都生成了filename_bb文件，所以可以直接得到所有程序的filename
    for line in file_list.readlines():
        file_list_new = re.findall(r'{}(.+?)_bb'.format(path),line)
        file_id_list.append(int(file_list_new[0])) #加入filename列表
    return file_id_list


def load_data(vocab):
    compiler_list=['gcc','llvm']
    # opti_tag_list=['O2','O3']
    # compiler_list=['gcc']
    opti_tag_list_gcc=['O0','O1','O2','O3','O0_ffast','O1_ffast','O2_ffast','O3_ffast']
    opti_tag_list_llvm=['O2','O3']
    poj_path='/workspace/poj_bench/'


    for compiler in compiler_list:
        if compiler == 'gcc':
            opti_tag_list = opti_tag_list_gcc
        else:
            opti_tag_list = opti_tag_list_llvm
        for opti_tag in opti_tag_list:
            in_path0=poj_path+'poj_data/'+compiler+'/'+opti_tag+'/'
            class_id_range=104 #debug 104
            for class_id in range(0,class_id_range): #debug: should be 104
                in_path=in_path0+str(class_id+1)+'/'
                file_list = get_file_in_poj_class(in_path) #读取这个文件夹下所有文件名
                for file_id in file_list:
                    print("LOAD DATA: class_id =",class_id,"file_id =",file_id,"compiler=",compiler,"opti_tag=",opti_tag)
                    load_ori(in_path+str(file_id),vocab)




if __name__ == '__main__':
    vocab = Vocab()
    load_data(vocab)

    #for bert
    vocab.save_cnt_dic()

    # #for out model
    # vocab.create_eb()
    # vocab.print_vocab()
    # vocab.save_dic()
    

