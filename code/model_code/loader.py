import torch.utils.data as data
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
import torch
import sparse as sp #version 0.1.0
import cfg
import numpy as np
from sklearn.decomposition import PCA
import time
import datetime
import glob
import pickle
import random
import pair_sample
import load_data
#import w2v
from collections import defaultdict
from torch.nn import functional as F
from torch import cuda

def init_print_list(): #初始化bianry相似性检测任务的pair
    pair_sample.init_print_list()

def print_memory_occupy(runtag): #输出内存占用情况
    print("[Memory]",runtag, cuda.memory_allocated(cfg.new_config.device1)/1024**2)
    
def time_list_print(time_list): #输出时间开销
    return
    if len(time_list)<2:
        return
    print('[Time LOADER]')
    for i in range(0,len(time_list)-1):
        print(round(time_list[i+1] - time_list[i],4),end=' ')
    print('')

def m_to_dense(ms): #稀疏矩阵转dense
    return ms.todense()
    #return ms

def coo_to_sparse_tensor(x): #本来尝试稀疏tensor，后发现不可行，还是转为普通tensor
    # print('before',x.coords.dtype,x.data.dtype,x.shape)
    # coords = x.coords
    # coords = coords.astype(int)
    # data = x.data
    # data = data.astype(float)
    # print('after',coords.dtype,data.dtype,x.shape)
    # return torch.sparse_coo_tensor(coords,data,x.shape)
    return torch.Tensor(x)

def to_tensor(x):
    return torch.Tensor(x)

def print_time(time_tag):
    print(time_tag,time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time())))

def resize_matrix(x,y,z,matrix): #调整矩阵size，保证每个样例的矩阵size相同
    if z<0:
        if len(matrix)>x:
            matrix = matrix[0:x]
        if len(matrix[0])>y:
            matrix = matrix[:,0:y]
        matrix = np.pad(matrix,((0,x-len(matrix)),(0,y-len(matrix[0]))),'constant',constant_values = (0,0))
        return matrix
    else:
        if len(matrix)>x:
            matrix = matrix[0:x]
        if len(matrix[0])>y:
            matrix = matrix[:,0:y]
        if len(matrix[0][0])>z:
            matrix = matrix[:,:,0:z]
        if x>len(matrix) or y>len(matrix[0]) or z>len(matrix[0][0]):
            print('[ERROR] resize,x,y,z too large')
        return matrix


class MyDataset(data.Dataset): #已废弃
    def __init__(self, CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map, label):
        self.CFG_As = CFG_As
        self.CFG_nodes = CFG_nodes
        self.cg_A = cg_A
        self.DFG_nodes = DFG_nodes
        self.dfg_A = dfg_A
        self.DFG_BB_map = DFG_BB_map
        self.CG_ginn_matrix = CG_ginn_matrix
        self.CG_ginn_node_map = CG_ginn_node_map
        self.DFG_ginn_matrix = DFG_ginn_matrix
        self.DFG_ginn_node_map = DFG_ginn_node_map
        self.label = label

    def __getitem__(self, index):#返回的是tensor
        CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map, label = self.CFG_As[index], self.CFG_nodes[index], self.cg_A[index], self.DFG_nodes[index], self.dfg_A[index], self.DFG_BB_map[index], self.CG_ginn_matrix[index], self.CG_ginn_node_map[index], self.DFG_ginn_matrix[index], self.DFG_ginn_node_map[index], self.label[index]
        
        CFG_As = m_to_dense(CFG_As)
        CFG_nodes = m_to_dense(CFG_nodes)
        cg_A = m_to_dense(cg_A)
        DFG_nodes = m_to_dense(DFG_nodes)
        dfg_A = m_to_dense(dfg_A)
        DFG_BB_map = m_to_dense(DFG_BB_map)
        CG_ginn_matrix = m_to_dense(CG_ginn_matrix)
        CG_ginn_node_map = m_to_dense(CG_ginn_node_map)
        DFG_ginn_matrix = m_to_dense(DFG_ginn_matrix)
        DFG_ginn_node_map = m_to_dense(DFG_ginn_node_map)
        label = label

        # if CFG_nodes.shape[3]>cfg.new_config.token_eb_dim or DFG_nodes.shape[2]>cfg.new_config.token_eb_dim:
        #     print("Error(token_eb_dim too small):",CFG_nodes.shape,DFG_nodes.shape,cfg.new_config.token_eb_dim)

        # CFG_nodes = np.pad(CFG_nodes,((0,0),(0,0),(0,0),(0,cfg.new_config.token_eb_dim-CFG_nodes.shape[3])),'constant',constant_values = (0,0))
        # DFG_nodes = np.pad(DFG_nodes,((0,0),(0,0),(0,cfg.new_config.token_eb_dim-DFG_nodes.shape[2])),'constant',constant_values = (0,0))
        
        # pca = PCA(n_components=cfg.new_config.new_token_eb_dim)
        
        # x1 = CFG_nodes.reshape(-1,cfg.new_config.token_eb_dim)
        # x1 = pca.fit_transform(x1)
        # CFG_nodes = x1.reshape(cfg.new_config.prog_func_max,cfg.new_config.func_bb_max,cfg.new_config.bb_token_max,cfg.new_config.new_token_eb_dim)

        # x1 = DFG_nodes.reshape(-1,cfg.new_config.token_eb_dim)
        # x1 = pca.fit_transform(x1)
        # DFG_nodes = x1.reshape(cfg.new_config.prog_inst_max,cfg.new_config.inst_token_max,cfg.new_config.new_token_eb_dim)

        
        return CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map, label


    def __len__(self):
        return len(self.label)

class NewDataset(data.Dataset): #已废弃
    def __init__(self, data_type):
        f_dir = cfg.new_config.pre_data_path +'/'+ data_type + '/' 
        self.data = glob.glob(f_dir+"*.pkl")
        
    def __getitem__(self, index):#返回的是tensor
        f_name = self.data[index]
        f = open(f_name, 'rb')
        dt = pickle.load(f)
        
        CFG_As = m_to_dense(dt['CFG_As'])
        CFG_nodes = m_to_dense(dt['CFG_nodes_low_dim'])
        cg_A = m_to_dense(dt['cg_A'])
        DFG_nodes = m_to_dense(dt['DFG_nodes_low_dim'])
        dfg_A = m_to_dense(dt['dfg_A'])
        DFG_BB_map = m_to_dense(dt['DFG_BB_map'])
        CG_ginn_matrix = m_to_dense(dt['CG_ginn_matrix'])
        CG_ginn_node_map = m_to_dense(dt['CG_ginn_node_map'])
        DFG_ginn_matrix = m_to_dense(dt['DFG_ginn_matrix'])
        DFG_ginn_node_map = m_to_dense(dt['DFG_ginn_node_map'])
        #CFG_nodes_low_dim = m_to_dense(dt['CFG_nodes_low_dim'])
        #DFG_nodes_low_dim = m_to_dense(dt['DFG_nodes_low_dim'])
        label = dt['label']

        CFG_As = torch.Tensor(CFG_As)
        CFG_nodes = torch.Tensor(CFG_nodes)
        cg_A = torch.Tensor(cg_A)
        DFG_nodes = torch.Tensor(DFG_nodes)
        dfg_A = torch.Tensor(dfg_A)
        DFG_BB_map = torch.Tensor(DFG_BB_map)
        CG_ginn_matrix = torch.Tensor(CG_ginn_matrix)
        CG_ginn_node_map = torch.Tensor(CG_ginn_node_map)
        DFG_ginn_matrix = torch.Tensor(DFG_ginn_matrix)
        DFG_ginn_node_map = torch.Tensor(DFG_ginn_node_map)
        label = torch.Tensor(label)
        
        #padding
        cfg_padding = CFG_As.permute(0,2,1)
        CFG_As = torch.cat((CFG_As,cfg_padding),2)

        cg_padding = cg_A.permute(1,0)
        cg_A = torch.cat((cg_A,cg_padding),1)

        dfg_padding = dfg_A.permute(1,0)
        dfg_A = torch.cat((dfg_A,dfg_padding),1)

        cg_ginn_padding = CG_ginn_matrix.permute(0,2,1)
        CG_ginn_matrix = torch.cat((CG_ginn_matrix,cg_ginn_padding),2)

        dfg_ginn_padding = DFG_ginn_matrix.permute(0,2,1)
        DFG_ginn_matrix = torch.cat((DFG_ginn_matrix,dfg_ginn_padding),2)
        
        return CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map, label


    def __len__(self):
        if cfg.new_config.debug:
            return 64 if 64<len(self.data) else len(self.data)
        else:
            return len(self.data)

class NewDataset2(data.Dataset): #已废弃
    def __init__(self, data_type):
        if cfg.new_config.pre_data_path != "/workspace/poj_bench/pre_data/all_sample0_random2/":
            self.data_type = data_type
            f_dir = cfg.new_config.pre_data_path +'/'
            #if cfg.new_config.task == 'funcClassify' and cfg.new_config.subtask == 'compile_class':
            #compile_class_tag_list=['0','1','4','5']
            
            compile_class_tag_list = cfg.new_config.compiler_tag_list.split(',')
            self.data = []
            for compile_class_tag in compile_class_tag_list:
                data_tmp = glob.glob(f_dir+"sample_"+compile_class_tag+"*.pkl")
                self.data = self.data + data_tmp
            
            # else:
            #     self.data = glob.glob(f_dir+"*.pkl")
            random.shuffle(self.data)
            train_amount = int(len(self.data) * 0.7)
            validation_amount = int(len(self.data) * 0.85)
            test_amount = len(self.data)
            if cfg.new_config.debug:
                train_amount=64
                validation_amount=128
                test_amount=192
            print("Dataset threshold:",train_amount,validation_amount,test_amount)
            if (data_type=='train'):
                self.data = self.data[0:train_amount]
            elif (data_type=='validation'):
                self.data = self.data[train_amount:validation_amount]
            else:
                self.data = self.data[validation_amount:test_amount]
        
        else:
            f_dir = cfg.new_config.pre_data_path +'/'+ data_type + '/' 
            self.data = glob.glob(f_dir+"*.pkl")
            print("Data length:",len(self.data),f_dir)
            if cfg.new_config.debug:
                self.data = self.data[:128]

        f_name = "/workspace/poj_bench/vocab_all_type.pkl"
        f = open(f_name, 'rb')
        dt = pickle.load(f)
        vocab_map = dt['map'] #{str,onehot_array}
        self.vocab = defaultdict(int)#{index,str}

        
  
    def __getitem__(self, index):#返回的是tensor
        f_name = self.data[index]
        f = open(f_name, 'rb')
        dt = pickle.load(f)
        
        CFG_As = m_to_dense(dt['CFG_As'])
        CFG_nodes = m_to_dense(dt['CFG_nodes_low_dim'])
        cg_A = m_to_dense(dt['cg_A'])
        DFG_nodes = m_to_dense(dt['DFG_nodes_low_dim'])
        dfg_A = m_to_dense(dt['dfg_A'])
        DFG_BB_map = m_to_dense(dt['DFG_BB_map'])
        CG_ginn_matrix = m_to_dense(dt['CG_ginn_matrix'])
        CG_ginn_node_map = m_to_dense(dt['CG_ginn_node_map'])
        DFG_ginn_matrix = m_to_dense(dt['DFG_ginn_matrix'])
        DFG_ginn_node_map = m_to_dense(dt['DFG_ginn_node_map'])
        #CFG_nodes_low_dim = m_to_dense(dt['CFG_nodes_low_dim'])
        #DFG_nodes_low_dim = m_to_dense(dt['DFG_nodes_low_dim'])
        #label = dt['label']
        if cfg.new_config.task=='funcClassify' and cfg.new_config.subtask=='compile_class':
            label = torch.Tensor(dt['label_compile'])
            if (label.size(0)<cfg.new_config.binary_class):
                label_padding = torch.zeros(cfg.new_config.binary_class-label.size(0))
                label = torch.cat((label,label_padding),0)
        else:
            label = torch.Tensor(dt['label_class'])
        #label_compile = torch.Tensor(dt['label_compile'])
        #label_bug = torch.Tensor(dt['label_bug'])


        #change one hot to w2v eb
        # #CFG_nodes[prog_func_max,func_bb_max,bb_token_max,token_eb_dim]
        # #DFG_nodes = [prog_inst_max,inst_token_max,token_eb_dim]
        # CFG_nodes = np.zeros((((cfg.new_config.prog_func_max,cfg.new_config.func_bb_max,cfg.new_config.new_token_eb_dim))))
        # DFG_nodes = np.zeros(((cfg.new_config.prog_inst_max,cfg.new_config.new_token_eb_dim)))
        
        # CFG_nodes0 = torch.Tensor(CFG_nodes0)
        # drop_t, CFG_nodes0_max_index = torch.max(CFG_nodes0,-1)
        # DFG_nodes0 = torch.Tensor(DFG_nodes0)
        # drop_t, DFG_nodes0_max_index = torch.max(DFG_nodes0,-1)
        # for i in range(cfg.new_config.prog_func_max):
        #     for j in range(cfg.new_config.func_bb_max):
        #         CFG_nodes[i,j] = self.w2v_trans(CFG_nodes0_max_index[i,j])
        # for i in range(cfg.new_config.prog_inst_max):
        #     DFG_nodes[i] = self.w2v_trans(DFG_nodes0_max_index[i])

        CFG_As = torch.Tensor(CFG_As)
        CFG_nodes = torch.Tensor(CFG_nodes)
        cg_A = torch.Tensor(cg_A)
        DFG_nodes = torch.Tensor(DFG_nodes)
        dfg_A = torch.Tensor(dfg_A)
        DFG_BB_map = torch.Tensor(DFG_BB_map)
        CG_ginn_matrix = torch.Tensor(CG_ginn_matrix)
        CG_ginn_node_map = torch.Tensor(CG_ginn_node_map)
        DFG_ginn_matrix = torch.Tensor(DFG_ginn_matrix)
        DFG_ginn_node_map = torch.Tensor(DFG_ginn_node_map)
        label = torch.Tensor(label)
        
        
        #padding
        cfg_padding = CFG_As.permute(0,2,1)
        CFG_As = torch.cat((CFG_As,cfg_padding),2)

        cg_padding = cg_A.permute(1,0)
        cg_A = torch.cat((cg_A,cg_padding),1)

        dfg_padding = dfg_A.permute(1,0)
        dfg_A = torch.cat((dfg_A,dfg_padding),1)

        cg_ginn_padding = CG_ginn_matrix.permute(0,2,1)
        CG_ginn_matrix = torch.cat((CG_ginn_matrix,cg_ginn_padding),2)

        dfg_ginn_padding = DFG_ginn_matrix.permute(0,2,1)
        DFG_ginn_matrix = torch.cat((DFG_ginn_matrix,dfg_ginn_padding),2)

        CG_ginn_node_map = CG_ginn_node_map[0]
        DFG_ginn_node_map = DFG_ginn_node_map[0]
        CG_ginn_matrix = CG_ginn_matrix[0]
        DFG_ginn_matrix = DFG_ginn_matrix[0]
        
        #print(CFG_As.shape, CFG_nodes.shape, cg_A.shape, DFG_nodes.shape, dfg_A.shape, DFG_BB_map.shape, CG_ginn_matrix.shape, CG_ginn_node_map.shape, DFG_ginn_matrix.shape, DFG_ginn_node_map.shape, label.shape)
        return CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map, label


    def __len__(self):
        if cfg.new_config.debug:
            return 64 if 64<len(self.data) else len(self.data)
        else:
            return len(self.data)

    def w2v_trans(self,index_matrix):
        #tokens[token_max,token_eb_dim]
        sentences = []
        index_list = list(index_matrix.numpy())
        for index in index_list:
            #index = np.where(t==np.max(t))[0][0]
            token = self.vocab[index]
            sentences.append(token)
        sen_vec = self.w2vModel.build_sentence_vector(sentences,cfg.new_config.new_token_eb_dim)
        return sen_vec

class NewDataset_wider(data.Dataset): #已废弃
    def __init__(self, data_type):
        if cfg.new_config.pre_data_path != "/workspace/poj_bench/pre_data/all_sample0_random2/":
            self.data_type = data_type
            f_dir = cfg.new_config.pre_data_path +'/'
            #if cfg.new_config.task == 'funcClassify' and cfg.new_config.subtask == 'compile_class':

            compile_class_tag_list = cfg.new_config.compiler_tag_list.split(',')
            #compile_class_tag_list=['0','1','4','5']
            
            self.data = []
            for compile_class_tag in compile_class_tag_list:
                data_tmp = glob.glob(f_dir+"sample_"+compile_class_tag+"*.pkl")
                self.data = self.data + data_tmp
                
            # else:
            #     self.data = glob.glob(f_dir+"*.pkl")
            random.shuffle(self.data)
            train_amount = int(len(self.data) * 0.7)
            validation_amount = int(len(self.data) * 0.85)
            test_amount = len(self.data)
            if cfg.new_config.debug:
                train_amount=256
                validation_amount=288
                test_amount=320
            print("Dataset threshold:",train_amount,validation_amount,test_amount)
            if (data_type=='train'):
                self.data = self.data[0:train_amount]
            elif (data_type=='validation'):
                self.data = self.data[train_amount:validation_amount]
            else:
                self.data = self.data[validation_amount:test_amount]
        
        else:
            f_dir = cfg.new_config.pre_data_path +'/'+ data_type + '/' 
            self.data = glob.glob(f_dir+"*.pkl")
            print("Data length:",len(self.data),f_dir)
            if cfg.new_config.debug:
                self.data = self.data[:128]

        # f_name = "/workspace/poj_bench/vocab_all_type.pkl"
        # f = open(f_name, 'rb')
        # dt = pickle.load(f)
        # vocab_map = dt['map'] #{str,onehot_array}
        # self.vocab = defaultdict(int)#{index,str}

        #change one hot to w2v eb
        # for key in vocab_map.keys():
        #     index = np.where(vocab_map[key]==np.max(vocab_map[key]))[0][0]
        #     self.vocab[index] = key
        #     #print('map:',index,key)
        # self.w2vModel = w2v.w2vModel()
        # save_path = '/workspace/NewModel/model_save/'
        # model_name = 'w2v.pkl'
        # self.w2vModel.load_model(save_path+model_name)

    def __getitem__(self, index):#返回的是tensor
        get_sample = False
        time1 = time.time()
        while get_sample==False:
            f_name = self.data[index]
            index += 1
            if index >= len(self.data):
                index = 0
            f = open(f_name, 'rb')
            dt = pickle.load(f)
            
            time2 = time.time()
            CFG_As = m_to_dense(dt['CFG_As']) 
            CFG_nodes = m_to_dense(dt['CFG_nodes_low_dim'])
            cg_A = m_to_dense(dt['cg_A'])
            DFG_nodes = m_to_dense(dt['DFG_nodes_low_dim'])
            dfg_A = m_to_dense(dt['dfg_A'])
            DFG_BB_map = None
            CG_ginn_matrix = None
            CG_ginn_node_map = None
            DFG_ginn_matrix = None
            DFG_ginn_node_map = None
            #CFG_nodes_low_dim = m_to_dense(dt['CFG_nodes_low_dim'])
            #DFG_nodes_low_dim = m_to_dense(dt['DFG_nodes_low_dim'])
            #label = dt['label']
            time3 = time.time()
            #print('Time load:',round(time2-time1,4),'Time to dense:',round(time3-time2,4),end=' ')
            if (cfg.new_config.func_bb_max < CFG_As.shape[-1] or \
                cfg.new_config.prog_func_max < CFG_As.shape[-3] or \
                cfg.new_config.bb_token_max < CFG_nodes.shape[-2] or \
                cfg.new_config.inst_token_max < DFG_nodes.shape[-2] or \
                cfg.new_config.prog_inst_max < DFG_nodes.shape[-3]):
                print("Loader: Next sample")
                print(cfg.new_config.func_bb_max,CFG_As.size(-1),\
                cfg.new_config.prog_func_max,CFG_As.size(-3),\
                cfg.new_config.bb_token_max,CFG_nodes.size(-2),\
                cfg.new_config.inst_token_max,DFG_nodes.size(-2),\
                cfg.new_config.prog_inst_max,DFG_nodes.size(-3))
                continue

            get_sample = True
            if cfg.new_config.task=='funcClassify' and cfg.new_config.subtask=='compile_class':
                label = torch.Tensor(dt['label_compile'])
                if (label.size(0)<cfg.new_config.binary_class):
                    label_padding = torch.zeros(cfg.new_config.binary_class-label.size(0))
                    label = torch.cat((label,label_padding),0)
            else:
                label = torch.Tensor(dt['label_class'])
            #label_compile = torch.Tensor(dt['label_compile'])
            #label_bug = torch.Tensor(dt['label_bug'])

            CFG_As = coo_to_sparse_tensor(CFG_As) #cg_A_node_size,func_bb_max,func_bb_max
            CFG_nodes = coo_to_sparse_tensor(CFG_nodes) #len(cg_A),func_bb_max,bb_token_max,new_token_eb_dim
            cg_A = coo_to_sparse_tensor(cg_A)
            DFG_nodes = coo_to_sparse_tensor(DFG_nodes) #len(dfg_A),inst_token_max,new_token_eb_dim
            dfg_A = coo_to_sparse_tensor(dfg_A)
            DFG_BB_map = torch.Tensor(1) #torch.Tensor(DFG_BB_map)
            CG_ginn_matrix = torch.Tensor(1) #torch.Tensor(CG_ginn_matrix)
            CG_ginn_node_map = torch.Tensor(1) #torch.Tensor(CG_ginn_node_map)
            DFG_ginn_matrix = torch.Tensor(1) #torch.Tensor(DFG_ginn_matrix)
            DFG_ginn_node_map = torch.Tensor(1) #torch.Tensor(DFG_ginn_node_map)
            label = torch.Tensor(label)

            # if (cfg.new_config.func_bb_max >= CFG_As.size(-1) and \
            #     cfg.new_config.prog_func_max >= CFG_As.size(-3) and \
            #     cfg.new_config.bb_token_max >= CFG_nodes.size(-2) and \
            #     cfg.new_config.inst_token_max >= DFG_nodes.size(-2) and \
            #     cfg.new_config.prog_inst_max >= DFG_nodes.size(-3)):
            #     get_sample = True
            #     # print(cfg.new_config.func_bb_max,CFG_As.size(-1),\
            #     # cfg.new_config.prog_func_max,CFG_As.size(-3),\
            #     # cfg.new_config.bb_token_max,CFG_nodes.size(-2),\
            #     # cfg.new_config.inst_token_max,DFG_nodes.size(-2),\
            #     # cfg.new_config.prog_inst_max,DFG_nodes.size(-3))
            # else:
            #     print('Dataloader: move to next sample:',CFG_As.shape,CFG_nodes.shape,DFG_nodes.shape)
            #     # print(cfg.new_config.func_bb_max,CFG_As.size(-1),\
            #     # cfg.new_config.prog_func_max,CFG_As.size(-3),\
            #     # cfg.new_config.bb_token_max,CFG_nodes.size(-2),\
            #     # cfg.new_config.inst_token_max,DFG_nodes.size(-2),\
            #     # cfg.new_config.prog_inst_max,DFG_nodes.size(-3))
      
        
        time4 = time.time()
            
        # #padding
        # cfg_0_padding = (0,cfg.new_config.func_bb_max - CFG_As.size(-1),\
        #                  0,cfg.new_config.func_bb_max - CFG_As.size(-2),\
        #                  0,cfg.new_config.prog_func_max - CFG_As.size(-3))
        # CFG_As = F.pad(CFG_As,cfg_0_padding)

        # cfg_nodes_0_padding = (0,0,\
        #                        0,cfg.new_config.bb_token_max - CFG_nodes.size(-2),\
        #                        0,cfg.new_config.func_bb_max - CFG_nodes.size(-3),\
        #                        0,cfg.new_config.prog_func_max - CFG_nodes.size(-4))
        # CFG_nodes = F.pad(CFG_nodes,cfg_nodes_0_padding)

        # cg_a_0_padding = (0,cfg.new_config.prog_func_max - cg_A.size(-1),\
        #                   0,cfg.new_config.prog_func_max - cg_A.size(-2))
        # cg_A = F.pad(cg_A,cg_a_0_padding)

        # dfg_nodes_0_padding = (0,0,\
        #                        0,cfg.new_config.inst_token_max - DFG_nodes.size(-2),\
        #                        0,cfg.new_config.prog_inst_max - DFG_nodes.size(-3))
        # DFG_nodes = F.pad(DFG_nodes,dfg_nodes_0_padding)

        # dfg_a_0_padding = (0,cfg.new_config.prog_inst_max - dfg_A.size(-1),\
        #                    0,cfg.new_config.prog_inst_max - dfg_A.size(-2))
        # dfg_A = F.pad(dfg_A,dfg_a_0_padding)

        # time5 =  time.time()
        #print('Time padding1:',round(time4-time3,4),end=' ')



        # cfg_padding = CFG_As.permute(0,2,1)
        # CFG_As = torch.cat((CFG_As,cfg_padding),2)

        # cg_padding = cg_A.permute(1,0)
        # cg_A = torch.cat((cg_A,cg_padding),1)

        # dfg_padding = dfg_A.permute(1,0)
        # dfg_A = torch.cat((dfg_A,dfg_padding),1)

        # time6 =  time.time()
        # #print('Time padding2:',round(time6-time4,4))

        '''
        # cg_ginn_padding = CG_ginn_matrix.permute(0,2,1)
        # CG_ginn_matrix = torch.cat((CG_ginn_matrix,cg_ginn_padding),2)

        # dfg_ginn_padding = DFG_ginn_matrix.permute(0,2,1)
        # DFG_ginn_matrix = torch.cat((DFG_ginn_matrix,dfg_ginn_padding),2)

        # CG_ginn_node_map = CG_ginn_node_map[0]
        # DFG_ginn_node_map = DFG_ginn_node_map[0]
        # CG_ginn_matrix = CG_ginn_matrix[0]
        # DFG_ginn_matrix = DFG_ginn_matrix[0]
        '''
        
        #print('[Time]:',round(time4-time1,3))
        #print(CFG_As.shape, CFG_nodes.shape, cg_A.shape, DFG_nodes.shape, dfg_A.shape, DFG_BB_map.shape, CG_ginn_matrix.shape, CG_ginn_node_map.shape, DFG_ginn_matrix.shape, DFG_ginn_node_map.shape, label.shape)
        #print(CFG_As.shape, CFG_nodes.shape, cg_A.shape, DFG_nodes.shape, dfg_A.shape)
        return CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map, label


    def __len__(self):
        if cfg.new_config.debug:
            return 256 if 256<len(self.data) else len(self.data) #64
        else:
            return 6000 if 6000<len(self.data) else len(self.data) 

class NewDataset_bert(data.Dataset): 
    def __init__(self, data_type, data_list): #data_type: train/test/validation data_list:所有程序样例的文件名列表
        self.data_type = data_type
        self.data = data_list
        
        #计算train/test/validation三个集合分别的样例数
        train_amount = int(len(self.data) * 0.7)
        validation_amount = int(len(self.data) * 0.85)
        test_amount = len(self.data)
        if len(self.data) >100000:
            train_amount = len(self.data)-20000
            validation_amount = len(self.data)-10000
            test_amount = len(self.data)
        if cfg.new_config.debug:
            train_amount=16
            validation_amount=32
            test_amount=40
        print("Dataset threshold :",train_amount,validation_amount,test_amount)
        if (data_type=='train'):
            self.data = self.data[0:train_amount] #self.data即为这个加载器所包含的数据，取数据就从self.data取
        elif (data_type=='validation'):
            self.data = self.data[train_amount:validation_amount]
        else:
            self.data = self.data[validation_amount:test_amount]
    

    def __getitem__(self, index): #返回的是tensor，此处的index值是在self.__len__ 返回的范围内的一个随机数，加载器通过__getitem__在self.data中获得位置为index的这个样例
        time_list = []
        time_list.append(time.time())
        f_name = self.data[index]
        f = open(f_name, 'rb')
        dt = pickle.load(f) #从文件读取这个数据

        class_id = torch.Tensor(dt['label_class']) #所属类别 1-104
        class_id = class_id.argmax(dim=0).long()
        #print('[Debug] loader: cg_A(class ',class_id,')\n', dt['cg_A'].coords)
        
        CFG_As = m_to_dense(dt['CFG_As']) 
        CFG_nodes = m_to_dense(dt['CFG_nodes_low_dim'])
        cg_A = m_to_dense(dt['cg_A'])
        DFG_nodes = m_to_dense(dt['DFG_nodes_low_dim'])
        dfg_A = m_to_dense(dt['dfg_A'])
        DFG_BB_map = None
        CG_ginn_matrix = None
        CG_ginn_node_map = None
        DFG_ginn_matrix = m_to_dense(dt['DFG_ginn_matrix'])
        DFG_ginn_node_map = m_to_dense(dt['DFG_ginn_node_map'])
        #CFG_nodes_low_dim = m_to_dense(dt['CFG_nodes_low_dim'])
        #DFG_nodes_low_dim = m_to_dense(dt['DFG_nodes_low_dim'])
        #label = dt['label']
        get_sample = True
        if cfg.new_config.task=='funcClassify' and cfg.new_config.subtask=='compile_class': #根据任务的不同决定label取什么
            label = torch.Tensor(dt['label_compile']) #编译类别
            # if (label.size(0)<cfg.new_config.binary_class):
            #     label_padding = torch.zeros(cfg.new_config.binary_class-label.size(0))
            #     label = torch.cat((label,label_padding),0)

            #print("dic:",cfg.new_config.compiler_dic)
            label0 = label.argmax(dim=-1)
            label0 = label0.item()
            label0 = cfg.new_config.compiler_dic[label0]
            label = torch.zeros(cfg.new_config.binary_class)
            label[label0] = 1 #修改编译类别这个tensor的维度，维度改为实际可能有的类别数
        else:
            label = torch.Tensor(dt['label_class'])
        #label_compile = torch.Tensor(dt['label_compile'])
        #label_bug = torch.Tensor(dt['label_bug'])
        #print('[DEBUG] lable',label,cfg.new_config.compiler_dic)
        time_list.append(time.time())

        CFG_As = to_tensor(CFG_As) #cg_A_node_size,func_bb_max,func_bb_max
        CFG_nodes = to_tensor(CFG_nodes) #len(cg_A),func_bb_max,bb_token_max,new_token_eb_dim
        cg_A = to_tensor(cg_A)
        DFG_nodes = to_tensor(DFG_nodes) #len(dfg_A),inst_token_max,new_token_eb_dim
        dfg_A = to_tensor(dfg_A)
        DFG_BB_map = torch.Tensor(1) #torch.Tensor(DFG_BB_map)
        CG_ginn_matrix = torch.Tensor(1) #torch.Tensor(CG_ginn_matrix)
        CG_ginn_node_map = torch.Tensor(1) #torch.Tensor(CG_ginn_node_map)
        DFG_ginn_matrix = to_tensor(DFG_ginn_matrix) #torch.Tensor(DFG_ginn_matrix)
        DFG_ginn_node_map = to_tensor(DFG_ginn_node_map) #torch.Tensor(DFG_ginn_node_map)
        label = torch.Tensor(label)


        # cfg_padding = CFG_As.permute(0,2,1)
        # CFG_As = torch.cat((CFG_As,cfg_padding),2)

        # cg_padding = cg_A.permute(1,0)
        # cg_A = torch.cat((cg_A,cg_padding),1)

        # dfg_padding = dfg_A.permute(1,0)
        # dfg_A = torch.cat((dfg_A,dfg_padding),1)

        # time6 =  time.time()
        # #print('Time padding2:',round(time6-time4,4))

        '''
        # cg_ginn_padding = CG_ginn_matrix.permute(0,2,1)
        # CG_ginn_matrix = torch.cat((CG_ginn_matrix,cg_ginn_padding),2)

        # dfg_ginn_padding = DFG_ginn_matrix.permute(0,2,1)
        # DFG_ginn_matrix = torch.cat((DFG_ginn_matrix,dfg_ginn_padding),2)

        # CG_ginn_node_map = CG_ginn_node_map[0]
        # DFG_ginn_node_map = DFG_ginn_node_map[0]
        # CG_ginn_matrix = CG_ginn_matrix[0]
        # DFG_ginn_matrix = DFG_ginn_matrix[0]
        '''
        
        time_list.append(time.time())
        time_list_print(time_list)
        #print('[Time]:',round(time4-time1,3))
        #print(CFG_As.shape, CFG_nodes.shape, cg_A.shape, DFG_nodes.shape, dfg_A.shape, DFG_BB_map.shape, CG_ginn_matrix.shape, CG_ginn_node_map.shape, DFG_ginn_matrix.shape, DFG_ginn_node_map.shape, label.shape)
        #print(CFG_As.shape, CFG_nodes.shape, cg_A.shape, DFG_nodes.shape, dfg_A.shape)
        return CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map, label


    def __len__(self): #返回数据集大小
        if cfg.new_config.debug:
            return 16 if 16<len(self.data) else len(self.data) #64
        else:
            #return 6000 if 6000<len(self.data) else len(self.data) 
            return len(self.data) 

class NewDataset_bert_pair(data.Dataset):
    def __init__(self, data_type):
        self.data_type = data_type
        pair_sample.from_file = cfg.new_config.pre_data_path

        #因为之前生成的程序样例数据都是单个存在，而binary相似性检测任务需要成对使用，所以要读取之前生成的【组队结果文件】，其中也标注了组队的两个样例是否同源
        if cfg.new_config.compiler_tag_list == '0,1':
            f_name = 'pair_data_set_list_llvm'
        elif cfg.new_config.compiler_tag_list == '4,5':
            f_name = 'pair_data_set_list_gcc'
        elif cfg.new_config.compiler_tag_list == '1,5':
            #f_name = 'pair_data_set_list_compiler_new'
            f_name = 'pair_data_set_list_compiler_new_all'
        else:
            f_name = 'pair_data_set_list_all'
        f = open(f_name, 'rb')
        dt = pickle.load(f)
        test_list = dt['test_list']
        validation_list = dt['validation_list']
        train_list = dt['train_list']
        pair_sample.filedic = dt['filedic']

        # if self.data_type == 'test':
        #     self.all_key_list = test_list
        # elif self.data_type == 'validation':
        #     self.all_key_list = validation_list
        # else:
        #     self.all_key_list = train_list

        self.all_key_list = train_list #实际上应该用上面注释的一段，因为之前生成的文件有问题，所以直接拿train_list来分为train/validation/test三部分了
        random.shuffle(self.all_key_list)
        train_amount = int(len(self.all_key_list) * 0.7)
        validation_amount = int(len(self.all_key_list) * 0.85)
        test_amount = len(self.all_key_list)
        if len(self.all_key_list) >100000:
            train_amount = len(self.all_key_list)-10000
            validation_amount = len(self.all_key_list)-5000
            test_amount = len(self.all_key_list)
        print("Dataset threshold :",train_amount,validation_amount,test_amount)
        if (data_type=='train'):
            self.all_key_list = self.all_key_list[0:train_amount]
        elif (data_type=='validation'):
            self.all_key_list = self.all_key_list[train_amount:validation_amount]
        else:
            self.all_key_list = self.all_key_list[validation_amount:test_amount]


        print('Pair dataloader load finish (%d samples)' %len(self.all_key_list))
    
    def __getitem__(self, index):
        dt_a, dt_b = pair_sample.load_with_all_key(self.all_key_list[index])

        CFG_As_a = torch.Tensor(m_to_dense(dt_a['CFG_As']))
        CFG_nodes_a = torch.Tensor(m_to_dense(dt_a['CFG_nodes_low_dim']))
        cg_A_a = torch.Tensor(m_to_dense(dt_a['cg_A']))
        DFG_nodes_a = torch.Tensor(m_to_dense(dt_a['DFG_nodes_low_dim']))
        dfg_A_a = torch.Tensor(m_to_dense(dt_a['dfg_A']))
        DFG_BB_map_a = torch.Tensor(1)
        CG_ginn_matrix_a = torch.Tensor(1)
        CG_ginn_node_map_a = torch.Tensor(1)
        DFG_ginn_matrix_a = torch.Tensor(1)
        DFG_ginn_node_map_a = torch.Tensor(1)
        #CFG_nodes_low_dim_a = torch.Tensor(m_to_dense(dt_a['CFG_nodes_low_dim']))
        #DFG_nodes_low_dim_a = torch.Tensor(m_to_dense(dt_a['DFG_nodes_low_dim']))
        label_class_a = torch.Tensor(dt_a['label_class'])
        label_compile_a = torch.Tensor(dt_a['label_compile'])
        label_bug_a = torch.Tensor(1)
        key_a = dt_a['key']

        CFG_As_b = torch.Tensor(m_to_dense(dt_b['CFG_As']))
        CFG_nodes_b = torch.Tensor(m_to_dense(dt_b['CFG_nodes_low_dim']))
        cg_A_b = torch.Tensor(m_to_dense(dt_b['cg_A']))
        DFG_nodes_b = torch.Tensor(m_to_dense(dt_b['DFG_nodes_low_dim']))
        dfg_A_b = torch.Tensor(m_to_dense(dt_b['dfg_A']))
        DFG_BB_map_b = torch.Tensor(1)
        CG_ginn_matrix_b = torch.Tensor(1)
        CG_ginn_node_map_b = torch.Tensor(1)
        DFG_ginn_matrix_b = torch.Tensor(1)
        DFG_ginn_node_map_b = torch.Tensor(1)
        #CFG_nodes_low_dim_b = torch.Tensor(m_to_dense(dt_b['CFG_nodes_low_dim']))
        #DFG_nodes_low_dim_b = torch.Tensor(m_to_dense(dt_b['DFG_nodes_low_dim']))
        label_class_b = torch.Tensor(dt_b['label_class'])
        label_compile_b = torch.Tensor(dt_b['label_compile'])
        label_bug_b = torch.Tensor(1)
        key_b = dt_b['key']

        if key_a == key_b:
            label = torch.Tensor([1])
        else:
            label = torch.Tensor([0])

        #print('[Debug]',label, key_a, key_b, label_class_a, label_class_b)


        
        # #padding
        # cfg_padding = CFG_As_a.permute(0,2,1)
        # CFG_As_a = torch.cat((CFG_As_a,cfg_padding),2)
        # cg_padding = cg_A_a.permute(1,0)
        # cg_A_a = torch.cat((cg_A_a,cg_padding),1)
        # dfg_padding = dfg_A_a.permute(1,0)
        # dfg_A_a = torch.cat((dfg_A_a,dfg_padding),1)
        # #cg_ginn_padding = CG_ginn_matrix_a.permute(0,2,1)
        # #CG_ginn_matrix_a = torch.cat((CG_ginn_matrix_a,cg_ginn_padding),2)
        # #dfg_ginn_padding = DFG_ginn_matrix_a.permute(0,2,1)
        # #DFG_ginn_matrix_a = torch.cat((DFG_ginn_matrix_a,dfg_ginn_padding),2)

        # cfg_padding = CFG_As_b.permute(0,2,1)
        # CFG_As_b = torch.cat((CFG_As_b,cfg_padding),2)
        # cg_padding = cg_A_b.permute(1,0)
        # cg_A_b = torch.cat((cg_A_b,cg_padding),1)
        # dfg_padding = dfg_A_b.permute(1,0)
        # dfg_A_b = torch.cat((dfg_A_b,dfg_padding),1)
        # #cg_ginn_padding = CG_ginn_matrix_b.permute(0,2,1)
        # #CG_ginn_matrix_b = torch.cat((CG_ginn_matrix_b,cg_ginn_padding),2)
        # #dfg_ginn_padding = DFG_ginn_matrix_b.permute(0,2,1)
        # #DFG_ginn_matrix_b = torch.cat((DFG_ginn_matrix_b,dfg_ginn_padding),2)

        # # #change the size 
        # # CG_ginn_node_map_a = CG_ginn_node_map_a[0]
        # # DFG_ginn_node_map_a = DFG_ginn_node_map_a[0]
        # # CG_ginn_node_map_b = CG_ginn_node_map_b[0]
        # # DFG_ginn_node_map_b = DFG_ginn_node_map_b[0]
        # # CG_ginn_matrix_a = CG_ginn_matrix_a[0]
        # # DFG_ginn_matrix_a = DFG_ginn_matrix_a[0]
        # # CG_ginn_matrix_b = CG_ginn_matrix_b[0]
        # # DFG_ginn_matrix_b = DFG_ginn_matrix_b[0]

        #print(CFG_As_a.shape, CFG_nodes_a.shape, cg_A_a.shape, DFG_nodes_a.shape, dfg_A_a.shape, DFG_BB_map_a.shape, CG_ginn_matrix_a.shape, CG_ginn_node_map_a.shape, DFG_ginn_matrix_a.shape, DFG_ginn_node_map_a.shape,\
        #           CFG_As_b.shape, CFG_nodes_b.shape, cg_A_b.shape, DFG_nodes_b.shape, dfg_A_b.shape, DFG_BB_map_b.shape, CG_ginn_matrix_b.shape, CG_ginn_node_map_b.shape, DFG_ginn_matrix_b.shape, DFG_ginn_node_map_b.shape, label)


        if cfg.new_config.task == 'funcClassify':
            print("[Error] wrong task!")
        elif cfg.new_config.task == 'binaryClassify':
            return CFG_As_a, CFG_nodes_a, cg_A_a, DFG_nodes_a, dfg_A_a, DFG_BB_map_a, CG_ginn_matrix_a, CG_ginn_node_map_a, DFG_ginn_matrix_a, DFG_ginn_node_map_a,\
                   CFG_As_b, CFG_nodes_b, cg_A_b, DFG_nodes_b, dfg_A_b, DFG_BB_map_b, CG_ginn_matrix_b, CG_ginn_node_map_b, DFG_ginn_matrix_b, DFG_ginn_node_map_b, label
        elif cfg.new_config.task == 'deadstore':
            print("[Error] wrong task!")
        else:
            print("[Error] wrong task!")
    

    def __len__(self):
        if cfg.new_config.debug:
            return 32 #if 64<len(self.all_key_list) else len(self.all_key_list)
        else:
            if len(self.all_key_list) > 200000:
                return 200000
            else:
                return len(self.all_key_list)

class DeadDataset(data.Dataset): #from all_sample0
    def __init__(self, data_type, data_list):
        self.data_type = data_type
        self.data = data_list
        
        train_amount = int(len(self.data) * 0.7)
        validation_amount = int(len(self.data) * 0.85)
        test_amount = len(self.data)
        if len(self.data) >100000:
            train_amount = len(self.data)-20000
            validation_amount = len(self.data)-10000
            test_amount = len(self.data)
        if cfg.new_config.debug:
            train_amount=16
            validation_amount=32
            test_amount=40
        print("Dataset threshold :",train_amount,validation_amount,test_amount)
        if (data_type=='train'):
            self.data = self.data[0:train_amount]
        elif (data_type=='validation'):
            self.data = self.data[train_amount:validation_amount]
        else:
            self.data = self.data[validation_amount:test_amount]
    

    def __getitem__(self, index):#返回的是tensor
        time_list = []
        time_list.append(time.time())
        f_name = self.data[index]
        f = open(f_name, 'rb')
        dt = pickle.load(f)
        #{'compile_tag':compile_tag,'bench_name':bench_name,'bench_id':bench_id,'dead_label':cfg_label,'cg_A':cg_A,'cg_all_nodes':cg_all_nodes,'cg_all_A':cg_all_A}

        #dead_label = torch.Tensor(dt['dead_label'])
        #print('[Debug] loader: cg_A(class ',class_id,')\n', dt['cg_A'].coords)
        
        #we need cg_A,CFG_nodes,CFG_As 
        CFG_As = m_to_dense(dt['cg_all_A']) 
        CFG_nodes = m_to_dense(dt['cg_all_nodes'])
        cg_A = m_to_dense(dt['cg_A'])
        DFG_nodes = None
        dfg_A = None
        DFG_BB_map = None 
        CG_ginn_matrix = None
        CG_ginn_node_map = None
        DFG_ginn_matrix = None
        DFG_ginn_node_map = None
        #CFG_nodes_low_dim = m_to_dense(dt['CFG_nodes_low_dim'])
        #DFG_nodes_low_dim = m_to_dense(dt['DFG_nodes_low_dim'])
        #label = dt['label']
        get_sample = True
        label = torch.zeros(2)
        label[dt['dead_label']] = 1
        time_list.append(time.time())


        CFG_As = resize_matrix(cfg.new_config.prog_func_max,cfg.new_config.func_bb_max,cfg.new_config.func_bb_max,CFG_As)
        CFG_nodes = resize_matrix(cfg.new_config.prog_func_max,cfg.new_config.func_bb_max,cfg.new_config.cfg_arg.init_dim,CFG_nodes)
        cg_A = resize_matrix(cfg.new_config.prog_func_max,cfg.new_config.prog_func_max,-1,cg_A)

        CFG_As = to_tensor(CFG_As) #cg_A_node_size,func_bb_max,func_bb_max
        CFG_nodes = to_tensor(CFG_nodes) #len(cg_A),func_bb_max,bb_token_max,new_token_eb_dim
        cg_A = to_tensor(cg_A)
        DFG_nodes = torch.Tensor(1) #len(dfg_A),inst_token_max,new_token_eb_dim
        dfg_A = torch.Tensor(1)
        DFG_BB_map = torch.Tensor(1) #torch.Tensor(DFG_BB_map)
        CG_ginn_matrix = torch.Tensor(1) #torch.Tensor(CG_ginn_matrix)
        CG_ginn_node_map = torch.Tensor(1) #torch.Tensor(CG_ginn_node_map)
        DFG_ginn_matrix = torch.Tensor(1) #torch.Tensor(DFG_ginn_matrix)
        DFG_ginn_node_map = torch.Tensor(1) #torch.Tensor(DFG_ginn_node_map)
        label = torch.Tensor(label)
        
        time_list.append(time.time())
        time_list_print(time_list)
        #print('[Time]:',round(time4-time1,3))
        #print(CFG_As.shape, CFG_nodes.shape, cg_A.shape, DFG_nodes.shape, dfg_A.shape, DFG_BB_map.shape, CG_ginn_matrix.shape, CG_ginn_node_map.shape, DFG_ginn_matrix.shape, DFG_ginn_node_map.shape, label.shape)
        #print(CFG_As.shape, CFG_nodes.shape, cg_A.shape, DFG_nodes.shape, dfg_A.shape)
        return CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map, label


    def __len__(self):
        if cfg.new_config.debug:
            return 16 if 16<len(self.data) else len(self.data) #64
        else:
            #return 6000 if 6000<len(self.data) else len(self.data) 
            return len(self.data) 


class NewDataset_binary_pair(data.Dataset):
    def __init__(self, data_type):
        '''
        create test/validation, random train set
        '''
        self.data_type = data_type
        f_dir = cfg.new_config.pre_data_path 

        compile_class_tag_list = cfg.new_config.compiler_tag_list.split(',')        
        self.data = []
        for compile_class_tag in compile_class_tag_list:
            data_tmp = glob.glob(f_dir+"sample_"+compile_class_tag+"*.pkl")
            self.data = self.data + data_tmp

        #self.data = glob.glob(f_dir+"*.pkl")
        if pair_sample.key_list==None:
            pair_sample.from_file = cfg.new_config.pre_data_path
            filedic = pair_sample.get_all_file_to_list(self.data)
            pair_sample.filedic = filedic
            pair_sample.key_list = list(filedic.keys())
        print("Dataset init:",len(pair_sample.key_list))

        test_amount = 1000
        if cfg.new_config.debug:
            test_amount = 32
        if self.data_type == 'test':
            self.all_key_list = pair_sample.create_set(test_amount,self.data_type)
        elif self.data_type == 'validation':
            self.all_key_list = pair_sample.create_set(test_amount,self.data_type)
        else:
            self.all_key_list = []
        print("Dataset init finish")

    
    def __getitem__(self, index):
        if self.data_type == 'test':
            dt_a, dt_b = pair_sample.load_with_all_key(self.all_key_list[index])
        elif self.data_type == 'validation':
            dt_a, dt_b = pair_sample.load_with_all_key(self.all_key_list[index])
        else:
            all_key = None
            while (all_key==None):
                typei = random.randint(0,1)
                if typei % 2 == 0:
                    all_key = pair_sample.create_false_pair(self.data_type)
                else:
                    all_key = pair_sample.create_true_pair(self.data_type)
            dt_a, dt_b = pair_sample.load_with_all_key(all_key)
            

        CFG_As_a = torch.Tensor(m_to_dense(dt_a['CFG_As']))
        CFG_nodes_a = torch.Tensor(m_to_dense(dt_a['CFG_nodes_low_dim']))
        cg_A_a = torch.Tensor(m_to_dense(dt_a['cg_A']))
        DFG_nodes_a = torch.Tensor(m_to_dense(dt_a['DFG_nodes_low_dim']))
        dfg_A_a = torch.Tensor(m_to_dense(dt_a['dfg_A']))
        DFG_BB_map_a = torch.Tensor(1)
        CG_ginn_matrix_a = torch.Tensor(1)
        CG_ginn_node_map_a = torch.Tensor(1)
        DFG_ginn_matrix_a = torch.Tensor(1)
        DFG_ginn_node_map_a = torch.Tensor(1)
        #CFG_nodes_low_dim_a = torch.Tensor(m_to_dense(dt_a['CFG_nodes_low_dim']))
        #DFG_nodes_low_dim_a = torch.Tensor(m_to_dense(dt_a['DFG_nodes_low_dim']))
        label_class_a = torch.Tensor(dt_a['label_class'])
        label_compile_a = torch.Tensor(dt_a['label_compile'])
        label_bug_a = torch.Tensor(1)
        key_a = dt_a['key']

        CFG_As_b = torch.Tensor(m_to_dense(dt_b['CFG_As']))
        CFG_nodes_b = torch.Tensor(m_to_dense(dt_b['CFG_nodes_low_dim']))
        cg_A_b = torch.Tensor(m_to_dense(dt_b['cg_A']))
        DFG_nodes_b = torch.Tensor(m_to_dense(dt_b['DFG_nodes_low_dim']))
        dfg_A_b = torch.Tensor(m_to_dense(dt_b['dfg_A']))
        DFG_BB_map_b = torch.Tensor(1)
        CG_ginn_matrix_b = torch.Tensor(1)
        CG_ginn_node_map_b = torch.Tensor(1)
        DFG_ginn_matrix_b = torch.Tensor(1)
        DFG_ginn_node_map_b = torch.Tensor(1)
        #CFG_nodes_low_dim_b = torch.Tensor(m_to_dense(dt_b['CFG_nodes_low_dim']))
        #DFG_nodes_low_dim_b = torch.Tensor(m_to_dense(dt_b['DFG_nodes_low_dim']))
        label_class_b = torch.Tensor(dt_b['label_class'])
        label_compile_b = torch.Tensor(dt_b['label_compile'])
        label_bug_b = torch.Tensor(1)
        key_b = dt_b['key']

        if key_a == key_b:
            label = 1
        else:
            label = 0

        # CFG_As = torch.Tensor(CFG_As)
        # CFG_nodes = torch.Tensor(CFG_nodes)
        # cg_A = torch.Tensor(cg_A)
        # DFG_nodes = torch.Tensor(DFG_nodes)
        # dfg_A = torch.Tensor(dfg_A)
        # DFG_BB_map = torch.Tensor(DFG_BB_map)
        # CG_ginn_matrix = torch.Tensor(CG_ginn_matrix)
        # CG_ginn_node_map = torch.Tensor(CG_ginn_node_map)
        # DFG_ginn_matrix = torch.Tensor(DFG_ginn_matrix)
        # DFG_ginn_node_map = torch.Tensor(DFG_ginn_node_map)
        # label_class = torch.Tensor(label_class)
        # label_compile = torch.Tensor(label_compile)
        # label_bug = torch.Tensor(label_bug)
        
        #padding
        cfg_padding = CFG_As_a.permute(0,2,1)
        CFG_As_a = torch.cat((CFG_As_a,cfg_padding),2)
        cg_padding = cg_A_a.permute(1,0)
        cg_A_a = torch.cat((cg_A_a,cg_padding),1)
        dfg_padding = dfg_A_a.permute(1,0)
        dfg_A_a = torch.cat((dfg_A_a,dfg_padding),1)
        #cg_ginn_padding = CG_ginn_matrix_a.permute(0,2,1)
        #CG_ginn_matrix_a = torch.cat((CG_ginn_matrix_a,cg_ginn_padding),2)
        #dfg_ginn_padding = DFG_ginn_matrix_a.permute(0,2,1)
        #DFG_ginn_matrix_a = torch.cat((DFG_ginn_matrix_a,dfg_ginn_padding),2)

        cfg_padding = CFG_As_b.permute(0,2,1)
        CFG_As_b = torch.cat((CFG_As_b,cfg_padding),2)
        cg_padding = cg_A_b.permute(1,0)
        cg_A_b = torch.cat((cg_A_b,cg_padding),1)
        dfg_padding = dfg_A_b.permute(1,0)
        dfg_A_b = torch.cat((dfg_A_b,dfg_padding),1)
        #cg_ginn_padding = CG_ginn_matrix_b.permute(0,2,1)
        #CG_ginn_matrix_b = torch.cat((CG_ginn_matrix_b,cg_ginn_padding),2)
        #dfg_ginn_padding = DFG_ginn_matrix_b.permute(0,2,1)
        #DFG_ginn_matrix_b = torch.cat((DFG_ginn_matrix_b,dfg_ginn_padding),2)

        # #change the size 
        # CG_ginn_node_map_a = CG_ginn_node_map_a[0]
        # DFG_ginn_node_map_a = DFG_ginn_node_map_a[0]
        # CG_ginn_node_map_b = CG_ginn_node_map_b[0]
        # DFG_ginn_node_map_b = DFG_ginn_node_map_b[0]
        # CG_ginn_matrix_a = CG_ginn_matrix_a[0]
        # DFG_ginn_matrix_a = DFG_ginn_matrix_a[0]
        # CG_ginn_matrix_b = CG_ginn_matrix_b[0]
        # DFG_ginn_matrix_b = DFG_ginn_matrix_b[0]

        #print(CFG_As_a.shape, CFG_nodes_a.shape, cg_A_a.shape, DFG_nodes_a.shape, dfg_A_a.shape, DFG_BB_map_a.shape, CG_ginn_matrix_a.shape, CG_ginn_node_map_a.shape, DFG_ginn_matrix_a.shape, DFG_ginn_node_map_a.shape,\
        #           CFG_As_b.shape, CFG_nodes_b.shape, cg_A_b.shape, DFG_nodes_b.shape, dfg_A_b.shape, DFG_BB_map_b.shape, CG_ginn_matrix_b.shape, CG_ginn_node_map_b.shape, DFG_ginn_matrix_b.shape, DFG_ginn_node_map_b.shape, label)


        if cfg.new_config.task == 'funcClassify':
            print("[Error] wrong task!")
        elif cfg.new_config.task == 'binaryClassify':
            return CFG_As_a, CFG_nodes_a, cg_A_a, DFG_nodes_a, dfg_A_a, DFG_BB_map_a, CG_ginn_matrix_a, CG_ginn_node_map_a, DFG_ginn_matrix_a, DFG_ginn_node_map_a,\
                   CFG_As_b, CFG_nodes_b, cg_A_b, DFG_nodes_b, dfg_A_b, DFG_BB_map_b, CG_ginn_matrix_b, CG_ginn_node_map_b, DFG_ginn_matrix_b, DFG_ginn_node_map_b, label
        elif cfg.new_config.task == 'bug':
            print("[Error] wrong task!")
        elif cfg.new_config.task == 'deadstore':
            print("[Error] wrong task!")
        else:
            print("[Error] wrong task!")
    

    def __len__(self):
        if cfg.new_config.debug:
            return 32 #if 64<len(self.all_key_list) else len(self.all_key_list)
        else:
            if self.data_type=='train':
                return 6000
            else:
                return len(self.all_key_list)

class NewDataset_binary_pair_pre(data.Dataset):
    def __init__(self, data_type):
        f_dir = cfg.new_config.pre_data_path +'/'+ data_type + '/' 
        self.data_a = sorted(glob.glob(f_dir+"*_a.pkl"))
        self.data_b = sorted(glob.glob(f_dir+"*_b.pkl"))


    def __getitem__(self, index):#返回的是tensor
        f_name = self.data_a[index]
        f = open(f_name, 'rb')
        dt_a = pickle.load(f)

        f_name = self.data_b[index]
        f = open(f_name, 'rb')
        dt_b = pickle.load(f)
        
        CFG_As_a = torch.Tensor(m_to_dense(dt_a['CFG_As']))
        CFG_nodes_a = torch.Tensor(m_to_dense(dt_a['CFG_nodes_low_dim']))
        cg_A_a = torch.Tensor(m_to_dense(dt_a['cg_A']))
        DFG_nodes_a = torch.Tensor(m_to_dense(dt_a['DFG_nodes_low_dim']))
        dfg_A_a = torch.Tensor(m_to_dense(dt_a['dfg_A']))
        DFG_BB_map_a = torch.Tensor(m_to_dense(dt_a['DFG_BB_map']))
        CG_ginn_matrix_a = torch.Tensor(m_to_dense(dt_a['CG_ginn_matrix']))
        CG_ginn_node_map_a = torch.Tensor(m_to_dense(dt_a['CG_ginn_node_map']))
        DFG_ginn_matrix_a = torch.Tensor(m_to_dense(dt_a['DFG_ginn_matrix']))
        DFG_ginn_node_map_a = torch.Tensor(m_to_dense(dt_a['DFG_ginn_node_map']))
        #CFG_nodes_low_dim_a = torch.Tensor(m_to_dense(dt_a['CFG_nodes_low_dim']))
        #DFG_nodes_low_dim_a = torch.Tensor(m_to_dense(dt_a['DFG_nodes_low_dim']))
        label_class_a = torch.Tensor(dt_a['label_class'])
        label_compile_a = torch.Tensor(dt_a['label_compile'])
        label_bug_a = torch.Tensor(dt_a['label_bug'])
        key_a = dt_a['key']

        CFG_As_b = torch.Tensor(m_to_dense(dt_b['CFG_As']))
        CFG_nodes_b = torch.Tensor(m_to_dense(dt_b['CFG_nodes_low_dim']))
        cg_A_b = torch.Tensor(m_to_dense(dt_b['cg_A']))
        DFG_nodes_b = torch.Tensor(m_to_dense(dt_b['DFG_nodes_low_dim']))
        dfg_A_b = torch.Tensor(m_to_dense(dt_b['dfg_A']))
        DFG_BB_map_b = torch.Tensor(m_to_dense(dt_b['DFG_BB_map']))
        CG_ginn_matrix_b = torch.Tensor(m_to_dense(dt_b['CG_ginn_matrix']))
        CG_ginn_node_map_b = torch.Tensor(m_to_dense(dt_b['CG_ginn_node_map']))
        DFG_ginn_matrix_b = torch.Tensor(m_to_dense(dt_b['DFG_ginn_matrix']))
        DFG_ginn_node_map_b = torch.Tensor(m_to_dense(dt_b['DFG_ginn_node_map']))
        #CFG_nodes_low_dim_b = torch.Tensor(m_to_dense(dt_b['CFG_nodes_low_dim']))
        #DFG_nodes_low_dim_b = torch.Tensor(m_to_dense(dt_b['DFG_nodes_low_dim']))
        label_class_b = torch.Tensor(dt_b['label_class'])
        label_compile_b = torch.Tensor(dt_b['label_compile'])
        label_bug_b = torch.Tensor(dt_b['label_bug'])
        key_b = dt_b['key']

        if key_a == key_b:
            label = 1
        else:
            label = 0

        # CFG_As = torch.Tensor(CFG_As)
        # CFG_nodes = torch.Tensor(CFG_nodes)
        # cg_A = torch.Tensor(cg_A)
        # DFG_nodes = torch.Tensor(DFG_nodes)
        # dfg_A = torch.Tensor(dfg_A)
        # DFG_BB_map = torch.Tensor(DFG_BB_map)
        # CG_ginn_matrix = torch.Tensor(CG_ginn_matrix)
        # CG_ginn_node_map = torch.Tensor(CG_ginn_node_map)
        # DFG_ginn_matrix = torch.Tensor(DFG_ginn_matrix)
        # DFG_ginn_node_map = torch.Tensor(DFG_ginn_node_map)
        # label_class = torch.Tensor(label_class)
        # label_compile = torch.Tensor(label_compile)
        # label_bug = torch.Tensor(label_bug)
        
        #padding
        cfg_padding = CFG_As_a.permute(0,2,1)
        CFG_As_a = torch.cat((CFG_As_a,cfg_padding),2)
        cg_padding = cg_A_a.permute(1,0)
        cg_A_a = torch.cat((cg_A_a,cg_padding),1)
        dfg_padding = dfg_A_a.permute(1,0)
        dfg_A_a = torch.cat((dfg_A_a,dfg_padding),1)
        cg_ginn_padding = CG_ginn_matrix_a.permute(0,2,1)
        CG_ginn_matrix_a = torch.cat((CG_ginn_matrix_a,cg_ginn_padding),2)
        dfg_ginn_padding = DFG_ginn_matrix_a.permute(0,2,1)
        DFG_ginn_matrix_a = torch.cat((DFG_ginn_matrix_a,dfg_ginn_padding),2)

        cfg_padding = CFG_As_b.permute(0,2,1)
        CFG_As_b = torch.cat((CFG_As_b,cfg_padding),2)
        cg_padding = cg_A_b.permute(1,0)
        cg_A_b = torch.cat((cg_A_b,cg_padding),1)
        dfg_padding = dfg_A_b.permute(1,0)
        dfg_A_b = torch.cat((dfg_A_b,dfg_padding),1)
        cg_ginn_padding = CG_ginn_matrix_b.permute(0,2,1)
        CG_ginn_matrix_b = torch.cat((CG_ginn_matrix_b,cg_ginn_padding),2)
        dfg_ginn_padding = DFG_ginn_matrix_b.permute(0,2,1)
        DFG_ginn_matrix_b = torch.cat((DFG_ginn_matrix_b,dfg_ginn_padding),2)

        #change the size 
        CG_ginn_node_map_a = CG_ginn_node_map_a[0]
        DFG_ginn_node_map_a = DFG_ginn_node_map_a[0]
        CG_ginn_node_map_b = CG_ginn_node_map_b[0]
        DFG_ginn_node_map_b = DFG_ginn_node_map_b[0]
        CG_ginn_matrix_a = CG_ginn_matrix_a[0]
        DFG_ginn_matrix_a = DFG_ginn_matrix_a[0]
        CG_ginn_matrix_b = CG_ginn_matrix_b[0]
        DFG_ginn_matrix_b = DFG_ginn_matrix_b[0]



        if cfg.new_config.task == 'funcClassify':
            print("[Error] wrong task!")
        elif cfg.new_config.task == 'binaryClassify':
            return CFG_As_a, CFG_nodes_a, cg_A_a, DFG_nodes_a, dfg_A_a, DFG_BB_map_a, CG_ginn_matrix_a, CG_ginn_node_map_a, DFG_ginn_matrix_a, DFG_ginn_node_map_a,\
                   CFG_As_b, CFG_nodes_b, cg_A_b, DFG_nodes_b, dfg_A_b, DFG_BB_map_b, CG_ginn_matrix_b, CG_ginn_node_map_b, DFG_ginn_matrix_b, DFG_ginn_node_map_b, label
        elif cfg.new_config.task == 'bug':
            print("[Error] wrong task!")
        elif cfg.new_config.task == 'deadstore':
            print("[Error] wrong task!")
        else:
            print("[Error] wrong task!")

       
        #return CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map, label


    def __len__(self):
        if cfg.new_config.debug:
            return 64 if 64<len(self.data_a) else len(self.data_a)
        else:
            return len(self.data_a)

class w2vDataset(data.Dataset): #from all_sample0
    def __init__(self, data_type):
        self.data_type = data_type
        self.data = load_data.loader_preprocessing()
        random.shuffle(self.data)
        train_amount = int(len(self.data) * 0.7)
        validation_amount = int(len(self.data) * 0.85)
        test_amount = len(self.data)
        if cfg.new_config.debug:
            train_amount=64
            validation_amount=128
            test_amount=192
        if (data_type=='train'):
            self.data = self.data[0:train_amount]
        elif (data_type=='validation'):
            self.data = self.data[train_amount:validation_amount]
        else:
            self.data = self.data[validation_amount:test_amount]

        
        # f_dir = cfg.new_config.pre_data_path +'/'+ data_type + '/' 
        # self.data = glob.glob(f_dir+"*.pkl")
        # print("Data length:",len(self.data),f_dir)
        # if cfg.new_config.debug:
        #     self.data = self.data[:128]
        
  
    def __getitem__(self, index):#返回的是tensor
        data_arg = self.data[index]
        
        dt = load_data.get_data_from_arg(data_arg)
        
        CFG_As = m_to_dense(dt['CFG_As'])
        CFG_nodes = m_to_dense(dt['CFG_nodes'])
        cg_A = m_to_dense(dt['cg_A'])
        DFG_nodes = m_to_dense(dt['DFG_nodes'])
        dfg_A = m_to_dense(dt['dfg_A'])
        DFG_BB_map = m_to_dense(dt['DFG_BB_map'])
        CG_ginn_matrix = m_to_dense(dt['CG_ginn_matrix'])
        CG_ginn_node_map = m_to_dense(dt['CG_ginn_node_map'])
        DFG_ginn_matrix = m_to_dense(dt['DFG_ginn_matrix'])
        DFG_ginn_node_map = m_to_dense(dt['DFG_ginn_node_map'])
        CFG_nodes_low_dim = m_to_dense(dt['CFG_nodes_low_dim'])
        DFG_nodes_low_dim = m_to_dense(dt['DFG_nodes_low_dim'])
        #label = dt['label']
        if cfg.new_config.task=='funcClassify' and cfg.new_config.subtask=='compile_class':
            label = torch.Tensor(dt['label_compile'])
            if (label.size(0)<cfg.new_config.binary_class):
                label_padding = torch.zeros(cfg.new_config.binary_class-label.size(0))
                label = torch.cat((label,label_padding),0)
        else:
            label = torch.Tensor(dt['label_class'])
        #label_compile = torch.Tensor(dt['label_compile'])
        #label_bug = torch.Tensor(dt['label_bug'])

        #CFG_nodes[prog_func_max,func_bb_max,bb_token_max,token_eb_dim]
        #DFG_nodes = [prog_inst_max,inst_token_max,token_eb_dim]




        CFG_As = torch.Tensor(CFG_As)
        CFG_nodes = torch.Tensor(CFG_nodes)
        cg_A = torch.Tensor(cg_A)
        DFG_nodes = torch.Tensor(DFG_nodes)
        dfg_A = torch.Tensor(dfg_A)
        DFG_BB_map = torch.Tensor(DFG_BB_map)
        CG_ginn_matrix = torch.Tensor(CG_ginn_matrix)
        CG_ginn_node_map = torch.Tensor(CG_ginn_node_map)
        DFG_ginn_matrix = torch.Tensor(DFG_ginn_matrix)
        DFG_ginn_node_map = torch.Tensor(DFG_ginn_node_map)
        label = torch.Tensor(label)
        
        
        #padding
        cfg_padding = CFG_As.permute(0,2,1)
        CFG_As = torch.cat((CFG_As,cfg_padding),2)

        cg_padding = cg_A.permute(1,0)
        cg_A = torch.cat((cg_A,cg_padding),1)

        dfg_padding = dfg_A.permute(1,0)
        dfg_A = torch.cat((dfg_A,dfg_padding),1)

        cg_ginn_padding = CG_ginn_matrix.permute(0,2,1)
        CG_ginn_matrix = torch.cat((CG_ginn_matrix,cg_ginn_padding),2)

        dfg_ginn_padding = DFG_ginn_matrix.permute(0,2,1)
        DFG_ginn_matrix = torch.cat((DFG_ginn_matrix,dfg_ginn_padding),2)

        CG_ginn_node_map = CG_ginn_node_map[0]
        DFG_ginn_node_map = DFG_ginn_node_map[0]
        CG_ginn_matrix = CG_ginn_matrix[0]
        DFG_ginn_matrix = DFG_ginn_matrix[0]
        
        #print(CFG_As.shape, CFG_nodes.shape, cg_A.shape, DFG_nodes.shape, dfg_A.shape, DFG_BB_map.shape, CG_ginn_matrix.shape, CG_ginn_node_map.shape, DFG_ginn_matrix.shape, DFG_ginn_node_map.shape, label)
        return CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map, label


    def __len__(self):
        if cfg.new_config.debug:
            return 64 if 64<len(self.data) else len(self.data)
        else:
            return len(self.data)

#数据加载器
class PrefetchDataLoader(DataLoader):
    '''
        replace DataLoader with PrefetchDataLoader
    '''
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


# class data_prefetcher(object):
#     def __init__(self, loader, device):
#         self.loader = loader
#         self.dataset = loader.dataset
#         self.stream = torch.cuda.Stream()
        
#         self.CFG_As = None
#         self.CFG_nodes = None
#         self.cg_A = None
#         self.DFG_nodes = None
#         self.dfg_A = None
#         self.DFG_BB_map = None
#         self.CG_ginn_matrix = None
#         self.CG_ginn_node_map = None
#         self.DFG_ginn_matrix = None
#         self.DFG_ginn_node_map = None
#         self.label = None

#         self.device = device

#     def __len__(self):
#         return len(self.loader)

#     def preload(self):
#         try:
#             self.CFG_As, self.CFG_nodes, self.cg_A, self.DFG_nodes, self.dfg_A, self.DFG_BB_map, self.CG_ginn_matrix, self.CG_ginn_node_map, self.DFG_ginn_matrix, self.DFG_ginn_node_map, self.label = next(self.loaditer)
#         except StopIteration:
#             self.CFG_As = None
#             self.CFG_nodes = None
#             self.cg_A = None
#             self.DFG_nodes = None
#             self.dfg_A = None
#             self.DFG_BB_map = None
#             self.CG_ginn_matrix = None
#             self.CG_ginn_node_map = None
#             self.DFG_ginn_matrix = None
#             self.DFG_ginn_node_map = None
#             self.label = None
#             return
#         with torch.cuda.stream(self.stream):
#             self.CFG_As = self.CFG_As.cuda(device=self.device, non_blocking=True)
#             self.CFG_nodes = self.CFG_nodes.cuda(device=self.device, non_blocking=True)
#             self.cg_A = self.cg_A.cuda(device=self.device, non_blocking=True)
#             self.DFG_nodes = self.DFG_nodes.cuda(device=self.device, non_blocking=True)
#             self.dfg_A = self.dfg_A.cuda(device=self.device, non_blocking=True)
#             self.DFG_BB_map = self.DFG_BB_map.cuda(device=self.device, non_blocking=True)
#             self.CG_ginn_matrix = self.CG_ginn_matrix.cuda(device=self.device, non_blocking=True)
#             self.CG_ginn_node_map = self.CG_ginn_node_map.cuda(device=self.device, non_blocking=True)
#             self.DFG_ginn_matrix = self.DFG_ginn_matrix.cuda(device=self.device, non_blocking=True)
#             self.DFG_ginn_node_map = self.DFG_ginn_node_map.cuda(device=self.device, non_blocking=True)
#             self.label = self.label.cuda(device=self.device, non_blocking=True)

#     def __iter__(self):
#         count = 0
#         self.loaditer = iter(self.loader)
#         self.preload()
#         while self.label is not None:
#             torch.cuda.current_stream().wait_stream(self.stream)

#             CFG_As = self.CFG_As
#             CFG_nodes = self.CFG_nodes
#             cg_A = self.cg_A
#             DFG_nodes = self.DFG_nodes
#             dfg_A = self.dfg_A
#             DFG_BB_map = self.DFG_BB_map
#             CG_ginn_matrix = self.CG_ginn_matrix
#             CG_ginn_node_map = self.CG_ginn_node_map
#             DFG_ginn_matrix = self.DFG_ginn_matrix
#             DFG_ginn_node_map = self.DFG_ginn_node_map
#             label = self.label

#             self.preload()
#             count += 1
#             yield CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map, label

#数据预取
class data_next_prefetcher(object): #TODO：不需要一一列出
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        
        self.CFG_As = None
        self.CFG_nodes = None
        self.cg_A = None
        self.DFG_nodes = None
        self.dfg_A = None
        self.DFG_BB_map = None
        self.CG_ginn_matrix = None
        self.CG_ginn_node_map = None
        self.DFG_ginn_matrix = None
        self.DFG_ginn_node_map = None
        self.label = None

        self.device = device
        self.preload()

    def preload(self):
        try:
            self.CFG_As, self.CFG_nodes, self.cg_A, self.DFG_nodes, self.dfg_A, self.DFG_BB_map, self.CG_ginn_matrix, self.CG_ginn_node_map, self.DFG_ginn_matrix, self.DFG_ginn_node_map, self.label = next(self.loader)
        except StopIteration:
            self.CFG_As = None
            self.CFG_nodes = None
            self.cg_A = None
            self.DFG_nodes = None
            self.dfg_A = None
            self.DFG_BB_map = None
            self.CG_ginn_matrix = None
            self.CG_ginn_node_map = None
            self.DFG_ginn_matrix = None
            self.DFG_ginn_node_map = None
            self.label = None
            return
        with torch.cuda.stream(self.stream): #加载到GPU
            self.CFG_As = self.CFG_As.cuda(device=self.device, non_blocking=True)
            self.CFG_nodes = self.CFG_nodes.cuda(device=self.device, non_blocking=True)
            self.cg_A = self.cg_A.cuda(device=self.device, non_blocking=True)
            self.DFG_nodes = self.DFG_nodes.cuda(device=self.device, non_blocking=True)
            self.dfg_A = self.dfg_A.cuda(device=self.device, non_blocking=True)
            self.DFG_BB_map = self.DFG_BB_map.cuda(device=self.device, non_blocking=True)
            self.CG_ginn_matrix = self.CG_ginn_matrix.cuda(device=self.device, non_blocking=True)
            self.CG_ginn_node_map = self.CG_ginn_node_map.cuda(device=self.device, non_blocking=True)
            self.DFG_ginn_matrix = self.DFG_ginn_matrix.cuda(device=self.device, non_blocking=True)
            self.DFG_ginn_node_map = self.DFG_ginn_node_map.cuda(device=self.device, non_blocking=True)
            self.label = self.label.cuda(device=self.device, non_blocking=True)

    def next(self): #训练过程中，每个batch通过next函数获取下一个batch的数据
        torch.cuda.current_stream().wait_stream(self.stream)
        #self.preload()
        CFG_As = self.CFG_As
        CFG_nodes = self.CFG_nodes
        cg_A = self.cg_A
        DFG_nodes = self.DFG_nodes
        dfg_A = self.dfg_A
        DFG_BB_map = self.DFG_BB_map
        CG_ginn_matrix = self.CG_ginn_matrix
        CG_ginn_node_map = self.CG_ginn_node_map
        DFG_ginn_matrix = self.DFG_ginn_matrix
        DFG_ginn_node_map = self.DFG_ginn_node_map
        label = self.label

        self.preload()

        return CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map, label

#pair数据预取
class data_next_prefetcher_pair(object): #TODO：可以考虑与data_next_prefetcher合并，也不需要一一列出
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        
        self.CFG_As_a = None
        self.CFG_nodes_a = None
        self.cg_A_a = None
        self.DFG_nodes_a = None
        self.dfg_A_a = None
        self.DFG_BB_map_a = None
        self.CG_ginn_matrix_a = None
        self.CG_ginn_node_map_a = None
        self.DFG_ginn_matrix_a = None
        self.DFG_ginn_node_map_a = None

        self.CFG_As_b = None
        self.CFG_nodes_b = None
        self.cg_A_b = None
        self.DFG_nodes_b = None
        self.dfg_A_b = None
        self.DFG_BB_map_b = None
        self.CG_ginn_matrix_b = None
        self.CG_ginn_node_map_b = None
        self.DFG_ginn_matrix_b = None
        self.DFG_ginn_node_map_b = None
        self.label = None

        self.device = device
        self.preload()

    def preload(self):
        try:
            self.CFG_As_a, self.CFG_nodes_a, self.cg_A_a, self.DFG_nodes_a, self.dfg_A_a, self.DFG_BB_map_a, self.CG_ginn_matrix_a, self.CG_ginn_node_map_a, self.DFG_ginn_matrix_a, self.DFG_ginn_node_map_a, \
            self.CFG_As_b, self.CFG_nodes_b, self.cg_A_b, self.DFG_nodes_b, self.dfg_A_b, self.DFG_BB_map_b, self.CG_ginn_matrix_b, self.CG_ginn_node_map_b, self.DFG_ginn_matrix_b, self.DFG_ginn_node_map_b, self.label = next(self.loader)
        except StopIteration:
            self.CFG_As_a = None
            self.CFG_nodes_a = None
            self.cg_A_a = None
            self.DFG_nodes_a = None
            self.dfg_A_a = None
            self.DFG_BB_map_a = None
            self.CG_ginn_matrix_a = None
            self.CG_ginn_node_map_a = None
            self.DFG_ginn_matrix_a = None
            self.DFG_ginn_node_map_a = None

            self.CFG_As_b = None
            self.CFG_nodes_b = None
            self.cg_A_b = None
            self.DFG_nodes_b = None
            self.dfg_A_b = None
            self.DFG_BB_map_b = None
            self.CG_ginn_matrix_b = None
            self.CG_ginn_node_map_b = None
            self.DFG_ginn_matrix_b = None
            self.DFG_ginn_node_map_b = None

            self.label = None
            return
        with torch.cuda.stream(self.stream):
            self.CFG_As_a = self.CFG_As_a.cuda(device=self.device, non_blocking=True)
            self.CFG_nodes_a = self.CFG_nodes_a.cuda(device=self.device, non_blocking=True)
            self.cg_A_a = self.cg_A_a.cuda(device=self.device, non_blocking=True)
            self.DFG_nodes_a = self.DFG_nodes_a.cuda(device=self.device, non_blocking=True)
            self.dfg_A_a = self.dfg_A_a.cuda(device=self.device, non_blocking=True)
            self.DFG_BB_map_a = self.DFG_BB_map_a.cuda(device=self.device, non_blocking=True)
            self.CG_ginn_matrix_a = self.CG_ginn_matrix_a.cuda(device=self.device, non_blocking=True)
            self.CG_ginn_node_map_a = self.CG_ginn_node_map_a.cuda(device=self.device, non_blocking=True)
            self.DFG_ginn_matrix_a = self.DFG_ginn_matrix_a.cuda(device=self.device, non_blocking=True)
            self.DFG_ginn_node_map_a = self.DFG_ginn_node_map_a.cuda(device=self.device, non_blocking=True)

            self.CFG_As_b = self.CFG_As_b.cuda(device=self.device, non_blocking=True)
            self.CFG_nodes_b = self.CFG_nodes_b.cuda(device=self.device, non_blocking=True)
            self.cg_A_b = self.cg_A_b.cuda(device=self.device, non_blocking=True)
            self.DFG_nodes_b = self.DFG_nodes_b.cuda(device=self.device, non_blocking=True)
            self.dfg_A_b = self.dfg_A_b.cuda(device=self.device, non_blocking=True)
            self.DFG_BB_map_b = self.DFG_BB_map_b.cuda(device=self.device, non_blocking=True)
            self.CG_ginn_matrix_b = self.CG_ginn_matrix_b.cuda(device=self.device, non_blocking=True)
            self.CG_ginn_node_map_b = self.CG_ginn_node_map_b.cuda(device=self.device, non_blocking=True)
            self.DFG_ginn_matrix_b = self.DFG_ginn_matrix_b.cuda(device=self.device, non_blocking=True)
            self.DFG_ginn_node_map_b = self.DFG_ginn_node_map_b.cuda(device=self.device, non_blocking=True)
            
            self.label = self.label

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        #self.preload()
        CFG_As_a = self.CFG_As_a
        CFG_nodes_a = self.CFG_nodes_a
        cg_A_a = self.cg_A_a
        DFG_nodes_a = self.DFG_nodes_a
        dfg_A_a = self.dfg_A_a
        DFG_BB_map_a = self.DFG_BB_map_a
        CG_ginn_matrix_a = self.CG_ginn_matrix_a
        CG_ginn_node_map_a = self.CG_ginn_node_map_a
        DFG_ginn_matrix_a = self.DFG_ginn_matrix_a
        DFG_ginn_node_map_a = self.DFG_ginn_node_map_a

        CFG_As_b = self.CFG_As_b
        CFG_nodes_b = self.CFG_nodes_b
        cg_A_b = self.cg_A_b
        DFG_nodes_b = self.DFG_nodes_b
        dfg_A_b = self.dfg_A_b
        DFG_BB_map_b = self.DFG_BB_map_b
        CG_ginn_matrix_b = self.CG_ginn_matrix_b
        CG_ginn_node_map_b = self.CG_ginn_node_map_b
        DFG_ginn_matrix_b = self.DFG_ginn_matrix_b
        DFG_ginn_node_map_b = self.DFG_ginn_node_map_b
        
        label = self.label

        self.preload()

        return CFG_As_a, CFG_nodes_a, cg_A_a, DFG_nodes_a, dfg_A_a, DFG_BB_map_a, CG_ginn_matrix_a, CG_ginn_node_map_a, DFG_ginn_matrix_a, DFG_ginn_node_map_a,\
            CFG_As_b, CFG_nodes_b, cg_A_b, DFG_nodes_b, dfg_A_b, DFG_BB_map_b, CG_ginn_matrix_b, CG_ginn_node_map_b, DFG_ginn_matrix_b, DFG_ginn_node_map_b, label
