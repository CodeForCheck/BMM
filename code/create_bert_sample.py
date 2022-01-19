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
#import sample as smp
from sample import prog
from sample import basicblock
from sample import function


from collections import defaultdict
import sparse as sp #version 0.1.0
import numpy as np
import pickle
import networkx as nx
import copy
import re
from sklearn.decomposition import PCA


def m_to_sparse(m):
    return sp.COO(m)

def m_to_dense(ms):
    return ms.todense()
    return ms

class basicblock():
    def __init__(self,addr,tokens,func_addr):
        self.addr = addr
        self.tokens = tokens
        self.func_addr = func_addr

class function():
    def __init__(self,addr,bb_list,cfg_A,):
        self.addr = addr
        self.bb_list = bb_list
        self.cfg_A = m_to_sparse(cfg_A)

class prog():
    def __init__(self,func_list,compile_tag,program_class,prog_bb_amount,prog_func_amount,file_id):
        self.func_list = func_list
        self.label_compile = compile_tag
        self.label_class = program_class
        self.label_fileid = file_id
        self.bb_amount = prog_bb_amount
        self.func_amount = prog_func_amount
    def save_sample(self,f_name):
        f = open(f_name + '.pkl', 'wb')
        sample_t = {'func_list':self.func_list,\
                    'bb_amount':self.bb_amount,\
                    'func_amount':self.func_amount,\
                    'label_fileid':self.label_fileid,\
                    'label_compile':self.label_compile,\
                    'label_class':self.label_class}
        pickle.dump(sample_t, f, pickle.HIGHEST_PROTOCOL)
       

    




special_ch=['all_token','null','*','[','(',')',':','-',']','_',',','+']
vocab_cnt_map = defaultdict(int)


parser = argparse.ArgumentParser()
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--sample_save_path', type=str, default='ori_sample2/')

def list_to_str(token_list):
    result = ''
    for t in token_list:
        result += t+' '
        if t not in vocab_cnt_map:
            vocab_cnt_map[t] = 1
        else:
            vocab_cnt_map[t] += 1
    return result

def get_sub_token_for_list(token_list):
    result=[]
    for token_s in token_list:
        result = result+get_sub_token(token_s)
    return result

def get_sub_token(token_s):
    t=''
    sub_token_list=[]
    for ch in token_s:
        if ch not in special_ch:
            t = t+ch
        else:
            if len(t)>0:
                sub_token_list.append(t)
            sub_token_list.append(ch)
            t=''
    if len(t)>0:
        sub_token_list.append(t)
    return sub_token_list


def load_ori(data_path,compile_tag,program_class,file_id):   #加载数据
        bb_f = open(data_path+"_bb")
        adj_f = open(data_path+"_adj",'rb')
        arg_f = open(data_path+"_arg")
        dfg_adj_f = open(data_path+"_dfg_adj",'rb')
        dfg_arg_f = open(data_path+"_dfg_arg")
        cg_adj_f = open(data_path+"_cg_adj",'rb')
        cg_arg_f = open(data_path+"_cg_arg")        
        if (os.path.getsize(data_path+"_bb")==0) or \
           (os.path.getsize(data_path+"_adj")==0) or \
           (os.path.getsize(data_path+"_arg")==0) or \
           (os.path.getsize(data_path+"_dfg_adj")==0) or \
           (os.path.getsize(data_path+"_dfg_arg")==0) or \
           (os.path.getsize(data_path+"_cg_adj")==0) or \
           (os.path.getsize(data_path+"_cg_arg")==0):
            return

        func_list = []
        prog_bb_amount = 0
        prog_func_amount = 0
        arg_line = arg_f.readline()
        while arg_line: #every func
            func_addr,bb_amount = map(int,arg_line.split())
            cfg_AS = np.load(adj_f,allow_pickle=True)[()]
            cfg_A = cfg_AS.toarray()
            arg_line = arg_f.readline()
            bb_ins_amount_list = list(map(int,arg_line.split()))
            bb_list = []
            prog_func_amount += 1
            for i in range(bb_amount): #every BB
                prog_bb_amount += 1
                bb_addr = -1
                tokens = '' #BB的tokens
                for j in range(bb_ins_amount_list[i]):
                    bb_line = bb_f.readline()
                    tmp_s=bb_line.split()
                    try:
                        inst_addr = int(tmp_s[0].rstrip(':'),16) #格式为addr: tokens 如（0x400578: sub rsp, 8）
                    except ValueError:
                        continue
                    if j==0:
                        bb_addr = inst_addr
                    del tmp_s[0]  #去除addr的部分，仅保留指令
                    tmp_s = list_to_str(get_sub_token_for_list(tmp_s)) #转为token序列
                    tokens += tmp_s #tmp_s为这条指令的tokens
                bb_list.append(basicblock(bb_addr,tokens,func_addr)) 

            func_list.append(function(func_addr,bb_list,cfg_A))
            arg_line = arg_f.readline()
            arg_line = arg_f.readline()
        sample_t = prog(func_list,compile_tag,program_class,prog_bb_amount,prog_func_amount,file_id)
        f_name = cfg.sample_save_path+'sample_'+str(compile_tag)+'_'+str(program_class)+'_'+str(file_id) #存储这个程序样例的路径
        sample_t.save_sample(f_name)#存储到文件
        

def get_file_in_poj_class(path):
    file_id_list=[]
    file_list = os.popen('ls '+path+'*_bb')
    #print(file_list)
    for line in file_list.readlines():
        file_list_new = re.findall(r'{}(.+?)_bb'.format(path),line)
        file_id_list.append(int(file_list_new[0]))
        #print(file_list_new[0])
    return file_id_list

def preprocessing():
    compiler_list=['gcc','llvm']
    opti_tag_list=['O2','O3']
    poj_path='/workspace/poj_bench/'
    
    for compiler in compiler_list:
        for opti_tag in opti_tag_list:
            in_path0=poj_path+'poj_data/'+compiler+'/'+opti_tag+'/'
            class_id_range=104
            if cfg.debug:
                class_id_range=2
            for class_id in range(0,class_id_range): #debug: should be 104
                in_path=in_path0+str(class_id+1)+'/'
                file_list = get_file_in_poj_class(in_path)
                for file_id in file_list:
                    if cfg.debug:
                        if file_id>50:
                            continue
                    print("LOAD DATA: class_id =",class_id,"file_id =",file_id,"compiler=",compiler,"opti_tag=",opti_tag)
                    if compiler=='gcc' and opti_tag=='O3':
                        binary_type=0
                    if compiler=='gcc' and opti_tag=='O2':
                        binary_type=1
                    if compiler=='llvm' and opti_tag=='O3':
                        binary_type=2
                    if compiler=='llvm' and opti_tag=='O2':
                        binary_type=3
                    load_ori(in_path+str(file_id),binary_type,class_id,file_id) #加载数据
                    if cfg.debug:
                        break


if __name__ == '__main__':
    cfg = parser.parse_args()
    preprocessing()
    
    # f_name = "vocab_cnt"
    # f = open(f_name + '.pkl', 'wb')
    # pickle.dump(vocab_cnt_map, f, pickle.HIGHEST_PROTOCOL)
