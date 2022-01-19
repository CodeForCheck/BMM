from collections import defaultdict
from random import choice
import pandas as pd
import numpy as np
import json
import cfg
import argparse
import torch
import torch.nn as nn
import random
from torch.autograd import Variable
import os
import math
import re
import time
import pickle
import threading
import sparse as sp #version 0.1.0
import networkx as nx


#这里的参数大部分都没有用，只是复用了之前的一些代码所以没有删除，仅pre_data_path需要设置为生成的数据的存储路径，以及最下面的6个参数需要修改为需要的值，其他参数的值没有意义
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default="1,2,3,4,5,6,7")
parser.add_argument('--pre_data_path', type=str, default="../poj_bench/pre_data/all_sample_all_type_bert32_smallDFG/") #数据存储路径
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--load_test', type=int, default=0)
parser.add_argument('--save_model', type=bool, default=True)
parser.add_argument('--model_type', type=str, default="all")
parser.add_argument('--token_eb_model', type=str, default="attention")
parser.add_argument('--lossfunc', type=str, default="bce")
parser.add_argument('--drop_rate', type=int, default=86)
parser.add_argument('--copy_rate', type=int, default=100)
parser.add_argument('--batch', type=int, default=4)
parser.add_argument('--target', type=str, default="o2_test")
parser.add_argument('--train', type=str, default="o2_test")
parser.add_argument('--model_tag', type=str, default="debug")
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=-1)
parser.add_argument('--task', type=str, default="binaryClassify", help='type of task(funcClassify,binaryClassify,bug,deadstore)')
parser.add_argument('--subtask', type=str, default="binaryClassify", help='type of subtask(task==binaryClassify)') 
parser.add_argument('--compiler_tag_list', type=str, default="0,1,2,3,4,5,6,7,8,9")
parser.add_argument('--token_eb_dim', type=int, default=1024, help='instruction token embedding size')
parser.add_argument('--new_token_eb_dim', type=int, default=128, help='small instruction token embedding size')
parser.add_argument('--inst_eb_dim', type=int, default=64, help='instruction embedding size')
parser.add_argument('--bb_eb_dim', type=int, default=64, help='bb embedding size')
parser.add_argument('--func_eb_dim', type=int, default=64, help='func embedding size')
parser.add_argument('--cfg_hidden_dim', type=int, default=128, help='GGNN hidden state size')
parser.add_argument('--cfg_init_dim', type=int, default=32, help='GGNN annotation_dim') #128
parser.add_argument('--cfg_out', type=int, default=128, help='GGNN output_dim')
parser.add_argument('--cfg_steps', type=int, default=10, help='cfg GGNN steps')
parser.add_argument('--cg_hidden_dim', type=int, default=128, help='GGNN hidden state size')
parser.add_argument('--cg_init_dim', type=int, default=128, help='GGNN annotation_dim')
parser.add_argument('--cg_out', type=int, default=128, help='GGNN output_dim')
parser.add_argument('--cg_steps', type=int, default=10, help='cfg GGNN steps')
parser.add_argument('--dfg_hidden_dim', type=int, default=128, help='GGNN hidden state size')
parser.add_argument('--dfg_init_dim', type=int, default=32, help='GGNN annotation_dim') #128
parser.add_argument('--dfg_out', type=int, default=128, help='GGNN output_dim')
parser.add_argument('--dfg_steps', type=int, default=10, help='cfg GGNN steps')
parser.add_argument('--inst_token_max', type=int, default=16, help='inst_token_max') #一条指令的token数上限
parser.add_argument('--bb_token_max', type=int, default=256, help='bb_token_max') #一个bb的token数上限
parser.add_argument('--func_bb_max', type=int, default=128, help='func_bb_max') #函数的bb数上限
parser.add_argument('--prog_func_max', type=int, default=64, help='prog_func_max') #程序的函数数上限
parser.add_argument('--prog_inst_max', type=int, default=512, help='prog_inst_max') #程序的指令数上限 #2048
parser.add_argument('--ginn_max_level', type=int, default=3, help='ginn_max_level') #ginn的收缩层数


class myThread (threading.Thread): #废弃
    def __init__(self, threadID, name, poj_path, binary_type, compiler, opti_tag):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.poj_path = poj_path
        self.binary_type = binary_type
        self.compiler = compiler
        self.opti_tag = opti_tag

    def run(self):
        print ("Start thread:" + self.name)
        in_path0=self.poj_path+'poj_data/'+self.compiler+'/'+self.opti_tag+'/'
        class_id_range=104 
        if cfg.new_config.debug:
            class_id_range=1
        for class_id in range(0,class_id_range): #debug: should be 0,104
            in_path=in_path0+str(class_id+1)+'/'
            file_list = get_file_in_poj_class(in_path)
            print('FILE_LIST_LENGTH:',len(file_list))
            for file_index, file_id in enumerate(file_list):
                print("LOAD DATA: class_id =",class_id,"file_id =",file_id,"compiler =",self.compiler,"opti_tag =",self.opti_tag)
                file_path = in_path+str(file_id)
                #load_ori(file_path, self.binary_type, class_id, file_id)

                load_ori(in_path+str(file_id),self.binary_type,class_id,file_id)
                if cfg.new_config.debug and file_index>=2:
                    print('[Debug] file_index>=2 break')
                    break
    
        print ("Finish thread:" + self.name)


class myThread_class (threading.Thread):
    def __init__(self, threadID, name, poj_path, class_id):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.poj_path = poj_path
        self.class_id = class_id

    def run(self):
        print ("Start thread:" + self.name)

        compiler_list=['llvm','gcc'] #['llvm','gcc']
        opti_tag_list_llvm=['O2','O3']
        opti_tag_list_gcc=['O2','O3']

        for compiler in compiler_list:
            if compiler == 'gcc':
                opti_tag_list = opti_tag_list_gcc
            else:
                opti_tag_list = opti_tag_list_llvm
            for opti_tag in opti_tag_list:
                if compiler=='llvm' and opti_tag=='O2':
                    binary_type=0
                if compiler=='llvm' and opti_tag=='O3':
                    binary_type=1
                if compiler=='gcc' and opti_tag=='O0':
                    binary_type=2
                if compiler=='gcc' and opti_tag=='O1':
                    binary_type=3
                if compiler=='gcc' and opti_tag=='O2':
                    binary_type=4
                if compiler=='gcc' and opti_tag=='O3':
                    binary_type=5
                if compiler=='gcc' and opti_tag=='O0_ffast':
                    binary_type=6
                if compiler=='gcc' and opti_tag=='O1_ffast':
                    binary_type=7
                if compiler=='gcc' and opti_tag=='O2_ffast':
                    binary_type=8
                if compiler=='gcc' and opti_tag=='O3_ffast':
                    binary_type=9

                in_path0=self.poj_path+'poj_data/'+compiler+'/'+opti_tag+'/'

                class_id = self.class_id
                in_path=in_path0+str(class_id+1)+'/'
                file_list = get_file_in_poj_class(in_path)
                print('FILE_LIST_LENGTH:',len(file_list))
                for file_index, file_id in enumerate(file_list):
                    print("LOAD DATA: class_id =",class_id,"file_id =",file_id,"compiler =",compiler,"opti_tag =",opti_tag, "(file_index",file_index,"/",len(file_list),")")
                    file_path = in_path+str(file_id)

                    load_ori(in_path+str(file_id),binary_type,class_id,file_id) #路径，编译类型，程序类别，程序编号
                    if cfg.new_config.debug and file_index>=2:
                        print('[Debug] file_index>=2 break')
                        break
    
        print ("Finish thread:" + self.name)

def m_to_sparse(m):
    return sp.COO(m)

def m_to_dense(ms):
    return ms.todense()

def dfg_remove(A, node_list, node_eb_matrix):
    new_node_list = list(set(node_list))
    new_A = np.zeros((len(new_node_list),len(new_node_list)))
    new_node_eb_matrix = np.zeros((len(new_node_list),len(node_eb_matrix[0])))
    addr_to_newi = {}
    oldi_to_newi = {}
    for i in range(len(new_node_list)): #every newi
        addr_to_newi[new_node_list[i]]=i
    
    all_change = np.zeros(len(new_node_list))
    for i in range(len(node_list)): #every oldi
        oldi_to_newi[i]=addr_to_newi[node_list[i]]
        new_node_eb_matrix[oldi_to_newi[i]]=node_eb_matrix[i]
        all_change[oldi_to_newi[i]]=1

    if all_change.sum() < len(all_change):
        print('[ERROR] wrong new_node_eb_matrix',all_change)
    
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i,j]==1:
                new_A[oldi_to_newi[i],oldi_to_newi[j]]=1
    return new_A, new_node_list, new_node_eb_matrix

class instruction():
    def __init__(self,addr,tokens):
        self.addr = addr
        if tokens == None:
            tokens = ["null"]
        self.tokens = tokens  #self.tokens + tokens
        self.eb = None


class basicblock():
    def __init__(self,addr,bb_inst_list,func_addr):
        self.addr = addr
        self.bb_inst_list = bb_inst_list
        self.func_addr = func_addr
        self.eb = None

class function():
    def __init__(self,addr,cfg_nodes,cfg_A,):
        self.addr = addr
        self.cfg_nodes = cfg_nodes
        self.cfg_A = cfg_A


class prog():
    def __init__(self,prog_inst_dic,func_list,compile_tag,program_class,dfg_A,cg_A,dfg_node_list,dfg_arg_bert_matrix,cg_node_list,file_id):
        self.prog_inst_dic = prog_inst_dic
        self.func_list = func_list
        self.compile_tag_label = compile_tag
        self.program_class_label = program_class
        self.dfg_A = dfg_A
        self.cg_A = cg_A
        self.dfg_node_list = dfg_node_list #other node: value = -1 (<SimProcedure __libc_start_main>)
        self.dfg_arg_bert_matrix = dfg_arg_bert_matrix
        self.cg_node_list = cg_node_list
        self.file_id = file_id
        self.valid = True

    #for ginn
    def get_x_window_adj_graph(self,A):
        I = np.identity(len(A))
        newMat = A+I
        oldMat = newMat
        flag = 0
        step = 1
        while (flag == 0) and (step < cfg.new_config.ginn_window):
            oldMat = newMat
            newMat = oldMat*(A+I)
            # for i in range(len(newMat)):
            #     for j in range(len(newMat)):
            #         if newMat[i, j] >= 1:
            #             newMat[i, j] = 1
            newMat = np.float32((newMat)>0)
            step += 1
            if (oldMat == newMat).all():
                flag = 1
        return newMat
        # relat_matrix = matrix
        # new_matrix = relat_matrix
        # old_matrix = new_matrix
        # reach_matrix = matrix
        # step = 0
        # unchange=False
        # while (unchange==False) and (step < cfg.new_config.ginn_window):
        #     #print('[DEBUG] get_x_window_adj_graph:get step',step,'matrix')
        #     old_matrix = new_matrix
        #     new_matrix = np.dot(old_matrix,relat_matrix) #old_matrix*relat_matrix
        #     new_matrix = np.int64(new_matrix>0)
        #     # for i in range(0,len(new_matrix)):
        #     #     for j in range(0,len(new_matrix)):
        #     #         if(new_matrix[i,j]>=1):
        #     #             new_matrix[i,j] = 1
        #     reach_matrix = np.int64((reach_matrix + new_matrix)>0)
        #     step = step+1
        #     if(old_matrix == new_matrix).all():
        #         unchange=True

        # return reach_matrix

    def get_sub_graph(self,adj_matrix,node_index,aggregation_graph_node, aggregation_sub_graph_node):
        for node in range(len(adj_matrix)):
            if node not in aggregation_graph_node:
                can_in = True
                for node_in in aggregation_sub_graph_node:
                    if adj_matrix[node,node_in]==0 and adj_matrix[node_in,node]==0: #node与子图中某个节点无关联
                        can_in = False
                if can_in:
                    aggregation_sub_graph_node.add(node)
        return aggregation_sub_graph_node

    def create_next_level_ginn_map(self,adj_matrix,node_num,level):
        #删除多余的0，减少计算规模
        maxNoneZeroI = max(np.nonzero(adj_matrix)[0])
        maxNoneZeroJ = max(np.nonzero(adj_matrix)[1])
        maxNoneZero = maxNoneZeroI if maxNoneZeroI>maxNoneZeroJ else maxNoneZeroJ
        adj_matrix = adj_matrix[:maxNoneZero+1,:maxNoneZero+1]

        #print('[DEBUG] None zero (level',level,')',len(adj_matrix.nonzero()[0]))
        w1_adj_matrix = adj_matrix
        w1_adj_matrix = np.pad(w1_adj_matrix,((0,node_num-len(w1_adj_matrix)),(0,node_num-len(w1_adj_matrix))),'constant',constant_values = (0,0))
        adj_matrix = self.get_x_window_adj_graph(adj_matrix)
        #print('[DEBUG] None zero (level',level,')',len(adj_matrix.nonzero()[0]))

        #[node_num,node_num]
        aggregation_graph = nx.DiGraph() #这一级的邻接矩阵
        aggregation_graph_node = set() #这一级已经处理过的节点
        ginn_map = np.zeros(((node_num,node_num,1))) #这一级聚集图和原图的映射关系
        node_index = 0
        aggr_node_index = 0
        #print('[DEBUG] create_next_level_ginn_map (level',level,'):',len(adj_matrix))
        while len(aggregation_graph_node) < len(adj_matrix): #还有节点未处理
            if node_index in aggregation_graph_node:
                node_index += 1
                continue

            aggregation_sub_graph_node = set()
            aggregation_sub_graph_node.add(node_index)
            aggregation_sub_graph_node = self.get_sub_graph(adj_matrix,node_index,aggregation_graph_node, aggregation_sub_graph_node)
            #print('[DEBUG] subgraph',node_index,len(aggregation_sub_graph_node),len(aggregation_graph_node),len(aggregation_graph))
            for node in aggregation_sub_graph_node:
                aggregation_graph_node.add(node)
                ginn_map[aggr_node_index,node] = [1] #[聚合节点,子节点]=1
            aggregation_graph.add_node(aggr_node_index)
            aggr_node_index += 1

        #aggregation_graph_matrix = np.zeros((len(aggregation_graph.nodes()),len(aggregation_graph.nodes())))
        aggregation_graph_matrix = np.zeros((node_num,node_num))
        ginn_map_matrix = ginn_map[:,:,0]
        # tmp_matrix = np.dot(np.transpose(ginn_map_matrix), w1_adj_matrix)
        # aggregation_graph_matrix = np.dot(tmp_matrix, ginn_map_matrix)
        trans_ginn_map_matrix = np.transpose(ginn_map_matrix)
        for i in range(len(aggregation_graph.nodes())):
            for j in range(len(adj_matrix)):
                if ginn_map_matrix[i,j]==1: 
                    for k in range(len(adj_matrix)):
                        if(adj_matrix[j,k]==1):
                            aggregation_graph_matrix[i][np.argmax(trans_ginn_map_matrix[k])]=1

        return aggregation_graph_matrix, ginn_map
               

    def create_ginn_map(self,matrix,node_num):  #[ginn_max_level,node_num,node_num]
        #先生成每一级，再分别补0 
        ginn_all_matrix = np.zeros((((cfg.new_config.ginn_max_level,node_num,node_num))))
        ginn_all_map = np.zeros((((cfg.new_config.ginn_max_level,node_num,node_num,1))))
        for i in range(cfg.new_config.ginn_max_level):
            # [node_num,node_num] 子图数，子图节点数，子图节点数
            next_adj_matrix, node_map = self.create_next_level_ginn_map(matrix,node_num,i)
            #next_adj_matrix补0
            ginn_all_matrix[i] = np.pad(next_adj_matrix,((0,node_num-len(next_adj_matrix)),(0,node_num-len(next_adj_matrix))),'constant',constant_values = (0,0))
            ginn_all_map[i] = node_map
            matrix = next_adj_matrix
        return ginn_all_matrix, ginn_all_map
          


    def create_all_task_data(self):
        CFG_As = np.zeros((cfg.new_config.prog_func_max,cfg.new_config.func_bb_max,cfg.new_config.func_bb_max))
        CFG_nodes = None 
        CFG_nodes_low_dim = np.zeros((cfg.new_config.prog_func_max,cfg.new_config.func_bb_max,cfg.new_config.cfg_arg.init_dim))
        DFG_BB_map = None

        if (len(self.cg_node_list) > cfg.new_config.prog_func_max) or (len(self.dfg_node_list) > cfg.new_config.prog_inst_max): #过滤数据，超过cfg，cg，dfg规模限制的数据不构造样例
            print('[Invalid] [',self.compile_tag_label,'] cg or dfg too large:',len(self.cg_node_list),len(self.dfg_node_list))
            self.valid = False
            return
        for i in range(len(self.cg_node_list)): #every func
            #print(len(self.func_list.keys()),len(self.cg_node_list))
            if self.cg_node_list[i] not in self.func_list.keys():
                print("[Error] func",self.cg_node_list[i],"not in func_list")
                #return None
                
            #将数据组织为设定大小的矩阵（不足的padding补0）
            func = self.func_list[self.cg_node_list[i]]
            func_cfg_a = np.pad(func.cfg_A,((0,cfg.new_config.func_bb_max-func.cfg_A.shape[0]),(0,cfg.new_config.func_bb_max-func.cfg_A.shape[1])),'constant',constant_values = (0,0))
            CFG_As[i] = func_cfg_a 
            func_cfg_nodes = np.pad(func.cfg_nodes,((0,cfg.new_config.func_bb_max-func.cfg_nodes.shape[0]),(0,cfg.new_config.cfg_arg.init_dim-func.cfg_nodes.shape[1])),'constant',constant_values = (0,0))
            CFG_nodes_low_dim[i] = func_cfg_nodes
            
        DFG_nodes = None 
        DFG_nodes_low_dim = np.zeros(((cfg.new_config.prog_inst_max,cfg.new_config.dfg_arg.init_dim)))
        
        if len(self.dfg_node_list) != (len(self.dfg_arg_bert_matrix)):
            print('[Error] The length of dfg_node_list & dfg_arg_bert_matrix are different:',self.dfg_node_list,self.dfg_arg_bert_matrix)
        for i in range(len(self.dfg_node_list)):
            DFG_nodes_low_dim[i] = self.dfg_arg_bert_matrix[i]

        CG_ginn_matrix, CG_ginn_node_map = None, None
        #DFG_ginn_matrix, DFG_ginn_node_map = None, None
        DFG_ginn_matrix, DFG_ginn_node_map = self.create_ginn_map(self.dfg_A,cfg.new_config.prog_inst_max) #[ginn_level_max,cfg.new_config.prog_inst_max,cfg.new_config.prog_inst_max]
        
        self.cg_A = np.pad(self.cg_A,((0,cfg.new_config.prog_func_max-self.cg_A.shape[0]),(0,cfg.new_config.prog_func_max-self.cg_A.shape[1])),'constant',constant_values = (0,0))
        self.dfg_A = np.pad(self.dfg_A,((0,cfg.new_config.prog_inst_max-self.dfg_A.shape[0]),(0,cfg.new_config.prog_inst_max-self.dfg_A.shape[1])),'constant',constant_values = (0,0))
        
        return_list = [CFG_As, CFG_nodes, self.cg_A, DFG_nodes, self.dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map, CFG_nodes_low_dim, DFG_nodes_low_dim]
        return return_list
                
    def create_all_task_sample(self):
        return_list = self.create_all_task_data()
        if return_list == None:
            return None
        else:
            label1 = np.zeros(cfg.new_config.func_class)
            label1[self.program_class_label] = 1
            return_list.append(label1)

            label2 = np.zeros(cfg.new_config.binary_class)
            label2[self.compile_tag_label] = 1
            return_list.append(label2)

            label3 = None
            return_list.append(label3)

            return_list.append(self.compile_tag_label)
            return_list.append(self.program_class_label)
            return_list.append(self.file_id)

            #class_label,compile_label,bug_label,cimpile_tag,class_tag,file_id
            return return_list

    def sample_save(self,new_sample_data):
        #class_label,compile_label,bug_label,compile_tag,class_tag,file_id
        sample_compile_type = new_sample_data[15]
        sample_class = new_sample_data[16]
        file_id = new_sample_data[17]
        #f_name = cfg.new_config.pre_data_path + "sample_" + str(sample_class) +'_'+ str(file_id) +'_'+ str(sample_compile_type)
        f_name = cfg.new_config.pre_data_path + "sample_" + str(sample_compile_type) +'_'+ str(file_id) +'_'+ str(sample_class)
        
        f = open(f_name + '.pkl', 'wb')
        result_list = new_sample_data
        CFG_As = m_to_sparse(result_list[0])
        CFG_nodes = None #m_to_sparse(result_list[1])
        cg_A = m_to_sparse(result_list[2])
        DFG_nodes = None #m_to_sparse(result_list[3])
        dfg_A = m_to_sparse(result_list[4])
        DFG_BB_map = None
        CG_ginn_matrix = None
        CG_ginn_node_map = None
        DFG_ginn_matrix = m_to_sparse(result_list[8])
        DFG_ginn_node_map = m_to_sparse(result_list[9])
        CFG_nodes_low_dim = m_to_sparse(result_list[10])
        DFG_nodes_low_dim = m_to_sparse(result_list[11])
        
        label_class = result_list[12]
        label_compile = result_list[13]
        label_bug = result_list[14]

        sample_t = {'CFG_As':CFG_As,\
                    'CFG_nodes':CFG_nodes,\
                    'cg_A':cg_A,\
                    'DFG_nodes':DFG_nodes,\
                    'dfg_A':dfg_A,\
                    'DFG_BB_map':DFG_BB_map,\
                    'CG_ginn_matrix':CG_ginn_matrix,\
                    'CG_ginn_node_map':CG_ginn_node_map,\
                    'DFG_ginn_matrix':DFG_ginn_matrix,\
                    'DFG_ginn_node_map':DFG_ginn_node_map,\
                    'CFG_nodes_low_dim':CFG_nodes_low_dim,\
                    'DFG_nodes_low_dim':DFG_nodes_low_dim,\
                    'label_compile':label_compile,\
                    'label_class':label_class,\
                    'label_bug':label_bug}
        pickle.dump(sample_t, f, pickle.HIGHEST_PROTOCOL)

def load_ori(data_path,compile_tag,program_class,file_id):   
    #os.path.exists('xxx/xxx/filename')
        
    if not ((os.path.exists(data_path+"_bb")) and \
        (os.path.exists(data_path+"_bb_bert")) and \
        (os.path.exists(data_path+"_adj")) and \
        (os.path.exists(data_path+"_arg")) and \
        (os.path.exists(data_path+"_dfg_adj")) and \
        (os.path.exists(data_path+"_dfg_arg")) and \
        (os.path.exists(data_path+"_dfg_bb_bert")) and \
        (os.path.exists(data_path+"_cg_adj")) and \
        (os.path.exists(data_path+"_cg_arg"))):
        print('[Invalid] [',compile_tag,'] lack of file')
        return
    bb_f = open(data_path+"_bb")
    bb_bert_f = open(data_path+"_bb_bert",'rb')
    adj_f = open(data_path+"_adj",'rb')
    arg_f = open(data_path+"_arg")
    dfg_adj_f = open(data_path+"_dfg_adj",'rb')
    dfg_arg_f = open(data_path+"_dfg_arg")
    dfg_arg_bert_f = open(data_path+"_dfg_bb_bert",'rb')
    cg_adj_f = open(data_path+"_cg_adj",'rb')
    cg_arg_f = open(data_path+"_cg_arg")     

    func_list = defaultdict(function)
    prog_inst_dic = defaultdict(instruction)
    prog_inst_dic[-1] = instruction(-1,None)
    arg_line = arg_f.readline()

    #加载cfg和dfg节点的初始embedding
    bb_bert_matrix = pickle.load(bb_bert_f)
    dfg_arg_bert_matrix = pickle.load(dfg_arg_bert_f)
    bb_bert_matrix = m_to_dense(bb_bert_matrix)
    dfg_arg_bert_matrix = m_to_dense(dfg_arg_bert_matrix)
    bb_bert_matrix_line = 0
    dfg_arg_bert_matrix_line = 0

    while arg_line: #every func
        func_addr,bb_amount = map(int,arg_line.split())
        cfg_AS = np.load(adj_f,allow_pickle=True)[()]
        cfg_A = cfg_AS.toarray() #CFG的邻接矩阵
        arg_line = arg_f.readline()
        bb_ins_amount_list = list(map(int,arg_line.split()))
        if bb_amount > cfg.new_config.func_bb_max: #函数的BB数过多，超过func_bb_max，则跳过这个样例
            print('[Invalid] [',compile_tag,'] func_bb_amount too large:',bb_amount)
            return
        cfg_nodes = np.zeros((cfg.new_config.func_bb_max,cfg.new_config.cfg_arg.init_dim)) #存储为一个固定大小的矩阵的形式
        for i in range(bb_amount):
            cfg_nodes[i] = bb_bert_matrix[bb_bert_matrix_line]
            bb_bert_matrix_line += 1
            
        func_list[func_addr]=function(func_addr,cfg_nodes,cfg_A)
        arg_line = arg_f.readline()
        arg_line = arg_f.readline()
    
    dfg_AS = np.load(dfg_adj_f,allow_pickle=True)[()] #dfg邻接矩阵
    dfg_A = dfg_AS.toarray()
    cg_AS = np.load(cg_adj_f,allow_pickle=True)[()] #cg邻接矩阵
    cg_A = cg_AS.toarray()
    dfg_node_amount = int(dfg_arg_f.readline())
    dfg_node_list = []
    for i in range(dfg_node_amount):
        node_addr = re.findall(r'<(.+?) id=',dfg_arg_f.readline()) #dfg_arg_f一行为一个节点，节点的格式为<0x400620 id=0x400620[2]> 我们需要的只有地址：0x400620 （[2]表示这是0x400620的2号节点，本来生成的dfg中，每条指令节点多余一个，去重了部分）
        if len(node_addr)>0:
            node_addr = int(node_addr[0],16)
            dfg_node_list.append(node_addr)
        else:
            dfg_node_list.append(-1) #some node in dfg is "invalid" node， 如<<SimProcedure __libc_start_main>>节点

    #去除dfg中的重复点（同一个指令会出现多次，因为调用不同，通过spec_dfg_build.py构造的数据去重过，但是通过cfg.py dfg.py cg.py构造的没去重过）
    len_ori_dfg_node_list = len(dfg_node_list)
    dfg_A, dfg_node_list, dfg_arg_bert_matrix = dfg_remove(dfg_A, dfg_node_list, dfg_arg_bert_matrix)
    print('[DEBUG] dfg_node_list_size',len_ori_dfg_node_list,'after dfg_remove',len(dfg_node_list))

    cg_node_amount = int(cg_arg_f.readline())
    cg_node_list = []
    for i in range(cg_node_amount):
        node_addr = int(cg_arg_f.readline())
        cg_node_list.append(node_addr)
    for i in range(cg_node_amount-1,-1,-1):
        if cg_node_list[i] not in func_list.keys(): #过滤掉没有cfg的cg节点
            del cg_node_list[i]
            cg_A = np.delete(cg_A,i,0)
            cg_A = np.delete(cg_A,i,1)
    
    new_sample = prog(prog_inst_dic,func_list,compile_tag,program_class,dfg_A,cg_A,dfg_node_list,dfg_arg_bert_matrix,cg_node_list,file_id) #数据全部存入prog类中
    new_sample_data = new_sample.create_all_task_sample() #对这个proj创造相应输入数据
    if new_sample.valid:
        new_sample.sample_save(new_sample_data) #将这个sample存储到文件
    # else:
    #     print('invalid '+str(compile_tag))
        


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
    compiler_list=['llvm','gcc']
    opti_tag_list_llvm=['O2','O3']
    opti_tag_list_gcc=['O0','O1','O2','O3','O0_ffast','O1_ffast','O2_ffast','O3_ffast']

    poj_path='../poj_bench/'


    threads = []
    #thread_amount = len(opti_tag_list_llvm) + len(opti_tag_list_gcc)

    # #poj_data(no bug)
    # for compiler in compiler_list:
    #     if compiler == 'gcc':
    #         opti_tag_list = opti_tag_list_gcc
    #     else:
    #         opti_tag_list = opti_tag_list_llvm
    #     for opti_tag in opti_tag_list:
    #         if compiler=='llvm' and opti_tag=='O2':
    #             binary_type=0
    #         if compiler=='llvm' and opti_tag=='O3':
    #             binary_type=1
    #         if compiler=='gcc' and opti_tag=='O0':
    #             binary_type=2
    #         if compiler=='gcc' and opti_tag=='O1':
    #             binary_type=3
    #         if compiler=='gcc' and opti_tag=='O2':
    #             binary_type=4
    #         if compiler=='gcc' and opti_tag=='O3':
    #             binary_type=5
    #         if compiler=='gcc' and opti_tag=='O0_ffast':
    #             binary_type=6
    #         if compiler=='gcc' and opti_tag=='O1_ffast':
    #             binary_type=7
    #         if compiler=='gcc' and opti_tag=='O2_ffast':
    #             binary_type=8
    #         if compiler=='gcc' and opti_tag=='O3_ffast':
    #             binary_type=9

    #         t_name = "Thread-"+str(binary_type)+"("+compiler+","+opti_tag+")"
    #         t = myThread(binary_type, t_name, poj_path, binary_type, compiler, opti_tag)
    #         threads.append(t)

    class_id_range=104 
    if cfg.new_config.debug:
        class_id_range=1
    thread_amount = class_id_range
    for class_id in range(0,class_id_range): # should be 0,104
        t_name = "Thread-"+str(class_id)
        t = myThread_class(class_id, t_name, poj_path, class_id)
        threads.append(t)


    
    for i in range(thread_amount):
        threads[i].start()

    for i in range(thread_amount):
        threads[i].join()
                

if __name__ == '__main__':
    print('Preparation start.')
    opt = parser.parse_args()
    config=cfg.CONFIG(opt)
    cfg.new_config = config
    
    if cfg.new_config.debug: #日志
        log_file = cfg.log_file('std','log/pre_data_'+time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time())))
    else:
        log_file = cfg.log_file('file','log/pre_data_'+time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time())))
    #log_file: 'file', 'std'
    cfg.new_config.print_c() #输出所生成的数据的参数(各个矩阵的size)

    preprocessing()
    print("Preprocessing finish.")
    log_file.close()
