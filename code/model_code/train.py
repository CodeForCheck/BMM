from collections import defaultdict
from random import choice
import pandas as pd
import numpy as np
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
from torch import cuda
import os
import math
import datetime
import time
import re
import glob
import ipdb

#import model_sparse as model
#import model_onedevice as model #iclr use
import model_ginn as model
import cfg
import load_data
import program
import loader
import mystatistics
#import databox as box


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default="7,6,5,4,3,2,1,0")
parser.add_argument('--pre_data_path', type=str, default="../poj_bench/pre_data/all_sample_all_type_bert32/")
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--load_test', type=str, default="null")
parser.add_argument('--save_model', type=bool, default=True)
parser.add_argument('--model_type', type=str, default="all")
parser.add_argument('--token_eb_model', type=str, default="bert")
parser.add_argument('--lossfunc', type=str, default="bce")
parser.add_argument('--drop_rate', type=int, default=86) #废弃
parser.add_argument('--copy_rate', type=int, default=100) #废弃
parser.add_argument('--batch', type=int, default=32) #max:32
parser.add_argument('--target', type=str, default="o2_test") #废弃
parser.add_argument('--train', type=str, default="o2_test") #废弃 
parser.add_argument('--model_tag', type=str, default="debug", help='tag of this experiment')
parser.add_argument('--epoch', type=int, default=1000, help='max epoch')
parser.add_argument('--lr', type=float, default=0.001) 
parser.add_argument('--weight_decay', type=float, default=-1)
parser.add_argument('--compiler_tag_list', type=str, default="0,1,2,3,4,5,6,7,8,9")
parser.add_argument('--task', type=str, default="funcClassify", help='type of task(funcClassify,binaryClassify,deadstore)')
parser.add_argument('--subtask', type=str, default="binaryPair", help='type of subtask(task==funcClassify)') #编译分类时选择compile_class，binary相似性检测时选择binaryPair，其他时候subtask本质无意义，但为了区分task=funcClassify时是程序分类还是编译分类，程序分类时设置为了binaryPair
parser.add_argument('--token_eb_dim', type=int, default=512, help='instruction token embedding size') #废弃
parser.add_argument('--new_token_eb_dim', type=int, default=32, help='small instruction token embedding size') #废弃
parser.add_argument('--inst_eb_dim', type=int, default=64, help='instruction embedding size') #废弃
parser.add_argument('--bb_eb_dim', type=int, default=64, help='bb embedding size') #废弃
parser.add_argument('--func_eb_dim', type=int, default=64, help='func embedding size') #废弃
parser.add_argument('--cfg_hidden_dim', type=int, default=128, help='GGNN hidden state size') #128
parser.add_argument('--cfg_init_dim', type=int, default=32, help='GGNN annotation_dim') #128
parser.add_argument('--cfg_out', type=int, default=128, help='GGNN output_dim')
parser.add_argument('--cfg_steps', type=int, default=8, help='cfg GGNN steps')
parser.add_argument('--cg_hidden_dim', type=int, default=128, help='GGNN hidden state size') 
parser.add_argument('--cg_init_dim', type=int, default=128, help='GGNN annotation_dim')
parser.add_argument('--cg_out', type=int, default=128, help='GGNN output_dim')
parser.add_argument('--cg_steps', type=int, default=4, help='cfg GGNN steps')
parser.add_argument('--dfg_hidden_dim', type=int, default=128, help='GGNN hidden state size') #128
parser.add_argument('--dfg_init_dim', type=int, default=32, help='GGNN annotation_dim') #128
parser.add_argument('--dfg_out', type=int, default=128, help='GGNN output_dim')
parser.add_argument('--dfg_steps', type=int, default=8, help='cfg GGNN steps')
parser.add_argument('--inst_token_max', type=int, default=16, help='inst_token_max') #一条指令的token数上限
parser.add_argument('--bb_token_max', type=int, default=256, help='bb_token_max') #一个bb的token数上限
parser.add_argument('--func_bb_max', type=int, default=128, help='func_bb_max') #函数的bb数上限
parser.add_argument('--prog_func_max', type=int, default=64, help='prog_func_max') #程序的函数数上限
parser.add_argument('--prog_inst_max', type=int, default=512, help='prog_inst_max') #程序的指令数上限
parser.add_argument('--ginn_max_level', type=int, default=2, help='ginn_max_level') #ginn的收缩层数
#末尾的6个参数需要与data_preparation_bert.py的设置保持一致 生成的数据是什么规模，加载的数据就是什么规模

# 16,128,32,16,4096
# xx,xx,256,32,1024
# func_class_sample2: 16,256,32,32,4096

# past to be 16,64,32,32,1024
# should be: 16,256,128,64,2048

def print_memory_occupy(runtag):
    return
    print("[Memory]",runtag, cuda.memory_allocated(cfg.new_config.device1)/1024**2)
    #print("  [Time]",time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time())))

def print_time(time_tag):
    print(time_tag,time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time())))

def detach_pre_tensor(tensor_list):
    for t in tensor_list:
        t.detach()

def time_list_print(time_list):
    if len(time_list)<2:
        return
    print('[Time TRAINER]')
    for i in range(0,len(time_list)-1):
        print(round(time_list[i+1] - time_list[i],4),end=' ')
    print('')

def model_run_pair(run_type,mf_model,epoch,new_data_loader): #已废弃
    # lr = cfg.new_config.lr * (0.1 ** (epoch // 15))
    # optimizer = torch.optim.Adam(mf_model.parameters(), lr=lr, weight_decay=0.001) 
    # criterion = model.ContrastiveLoss().to(cfg.new_config.device_label)
    
    st = mystatistics.Statistics(cfg.new_config.task,cfg.new_config.subtask)
    total_loss=0
    total_loss_cnt=0
    iter_i=0
       
    # debug:another way: this for should in loop before: [for part_i in range(part_amount):]
    for CFG_As_a, CFG_nodes_a, cg_A_a, DFG_nodes_a, dfg_A_a, DFG_BB_map_a, CG_ginn_matrix_a, CG_ginn_node_map_a, DFG_ginn_matrix_a, DFG_ginn_node_map_a,\
                   CFG_As_b, CFG_nodes_b, cg_A_b, DFG_nodes_b, dfg_A_b, DFG_BB_map_b, CG_ginn_matrix_b, CG_ginn_node_map_b, DFG_ginn_matrix_b, DFG_ginn_node_map_b, label  in new_data_loader: 
        print_time("New batch start")

        CFG_As_a = Variable(CFG_As_a).float()
        CFG_nodes_a = Variable(CFG_nodes_a).float()
        cg_A_a = Variable(cg_A_a).float()
        DFG_nodes_a = Variable(DFG_nodes_a).float()
        dfg_A_a = Variable(dfg_A_a).float()
        DFG_BB_map_a = Variable(DFG_BB_map_a).float()
        CG_ginn_matrix_a = Variable(CG_ginn_matrix_a).float()
        CG_ginn_node_map_a = Variable(CG_ginn_node_map_a).float()
        DFG_ginn_matrix_a = Variable(DFG_ginn_matrix_a).float()
        DFG_ginn_node_map_a = Variable(DFG_ginn_node_map_a).float()

        CFG_As_b = Variable(CFG_As_b).float()
        CFG_nodes_b = Variable(CFG_nodes_b).float()
        cg_A_b = Variable(cg_A_b).float()
        DFG_nodes_b = Variable(DFG_nodes_b).float()
        dfg_A_b = Variable(dfg_A_b).float()
        DFG_BB_map_b = Variable(DFG_BB_map_b).float()
        CG_ginn_matrix_b = Variable(CG_ginn_matrix_b).float()
        CG_ginn_node_map_b = Variable(CG_ginn_node_map_b).float()
        DFG_ginn_matrix_b = Variable(DFG_ginn_matrix_b).float()
        DFG_ginn_node_map_b = Variable(DFG_ginn_node_map_b).float()
        
        label = Variable(label).float().to(cfg.new_config.device_label)

        #print_time("Start train")
        output1,output2 = mf_model(CFG_As_a, CFG_nodes_a, cg_A_a, DFG_nodes_a, dfg_A_a, DFG_BB_map_a, CG_ginn_matrix_a, CG_ginn_node_map_a, DFG_ginn_matrix_a, DFG_ginn_node_map_a, \
                                        CFG_As_b, CFG_nodes_b, cg_A_b, DFG_nodes_b, dfg_A_b, DFG_BB_map_b, CG_ginn_matrix_b, CG_ginn_node_map_b, DFG_ginn_matrix_b, DFG_ginn_node_map_b)
        #print_time("Forward finish")
        #loss = criterion(torch.max(output,-1)[1].float(), label)
        loss = criterion(output1,output2,label)
        total_loss += loss.item()
        total_loss_cnt += 1
        #print_time("Loss cnt finish")
        #print("loss:",loss.item())
        
        if run_type == 'train':
            if(iter_i % 5) == 0:
                print("iter "+str(iter_i)+": total loss ",total_loss/total_loss_cnt)
            mf_model.zero_grad()
            loss.backward()
            optimizer.step()
            #print_time("Backward")
        else:
            euclidean_distance = F.pairwise_distance(output1, output2) #计算距离
            euclidean_distance = euclidean_distance.detach().cpu().numpy().tolist()
            label = label.cpu().numpy().tolist()
            print(label,euclidean_distance)
            
            st.add_cnt(euclidean_distance,label,run_type)
        iter_i += 1

    if run_type == 'train':
        print("train epoch_loss:",total_loss/total_loss_cnt)


    stop=False
    if run_type != 'train':
        #print("validation----------------------------")
        print("validation epoch_loss,min_loss:",total_loss/total_loss_cnt,cfg.new_config.validation_loss_min)
        #print("---------------------------------------")
        stop = st.print_epoch_reuslt(total_loss/total_loss_cnt,run_type)

    return stop

def model_run_pair_prefetcher(run_type,mf_model,epoch,new_data_loader):
    # lr = cfg.new_config.lr * (0.1 ** (epoch // 15))
    # optimizer = torch.optim.Adam(mf_model.parameters(), lr=lr, weight_decay=0.001) 
    # criterion = model.ContrastiveLoss().to(cfg.new_config.device_label)

    prefetcher = loader.data_next_prefetcher_pair(new_data_loader,cfg.new_config.device1) #将loader包装为预加载器
    st = mystatistics.Statistics(cfg.new_config.task,cfg.new_config.subtask)
    total_loss=0
    total_loss_cnt=0
    iter_i=0
       
    #可以优化，不需要列这么多，下面的.float()也可以通过循环处理
    CFG_As_a, CFG_nodes_a, cg_A_a, DFG_nodes_a, dfg_A_a, DFG_BB_map_a, CG_ginn_matrix_a, CG_ginn_node_map_a, DFG_ginn_matrix_a, DFG_ginn_node_map_a,\
        CFG_As_b, CFG_nodes_b, cg_A_b, DFG_nodes_b, dfg_A_b, DFG_BB_map_b, CG_ginn_matrix_b, CG_ginn_node_map_b, DFG_ginn_matrix_b, DFG_ginn_node_map_b, label = prefetcher.next() #从预加载器拿一个batch的数据
    # for CFG_As_a, CFG_nodes_a, cg_A_a, DFG_nodes_a, dfg_A_a, DFG_BB_map_a, CG_ginn_matrix_a, CG_ginn_node_map_a, DFG_ginn_matrix_a, DFG_ginn_node_map_a,\
    #                CFG_As_b, CFG_nodes_b, cg_A_b, DFG_nodes_b, dfg_A_b, DFG_BB_map_b, CG_ginn_matrix_b, CG_ginn_node_map_b, DFG_ginn_matrix_b, DFG_ginn_node_map_b, label  in new_data_loader: 

    while label is not None:
        #TODO：其实可以通过循环来处理，不需要写这么多
        CFG_As_a = Variable(CFG_As_a).float()
        CFG_nodes_a = Variable(CFG_nodes_a).float()
        cg_A_a = Variable(cg_A_a).float()
        DFG_nodes_a = Variable(DFG_nodes_a).float()
        dfg_A_a = Variable(dfg_A_a).float()
        DFG_BB_map_a = Variable(DFG_BB_map_a).float()
        CG_ginn_matrix_a = Variable(CG_ginn_matrix_a).float()
        CG_ginn_node_map_a = Variable(CG_ginn_node_map_a).float()
        DFG_ginn_matrix_a = Variable(DFG_ginn_matrix_a).float()
        DFG_ginn_node_map_a = Variable(DFG_ginn_node_map_a).float()

        CFG_As_b = Variable(CFG_As_b).float()
        CFG_nodes_b = Variable(CFG_nodes_b).float()
        cg_A_b = Variable(cg_A_b).float()
        DFG_nodes_b = Variable(DFG_nodes_b).float()
        dfg_A_b = Variable(dfg_A_b).float()
        DFG_BB_map_b = Variable(DFG_BB_map_b).float()
        CG_ginn_matrix_b = Variable(CG_ginn_matrix_b).float()
        CG_ginn_node_map_b = Variable(CG_ginn_node_map_b).float()
        DFG_ginn_matrix_b = Variable(DFG_ginn_matrix_b).float()
        DFG_ginn_node_map_b = Variable(DFG_ginn_node_map_b).float()
        
        label = Variable(label).float().to(cfg.new_config.device_label)

        #print_time("Start train")
        output1,output2 = mf_model(CFG_As_a, CFG_nodes_a, cg_A_a, DFG_nodes_a, dfg_A_a, DFG_BB_map_a, CG_ginn_matrix_a, CG_ginn_node_map_a, DFG_ginn_matrix_a, DFG_ginn_node_map_a, \
                                        CFG_As_b, CFG_nodes_b, cg_A_b, DFG_nodes_b, dfg_A_b, DFG_BB_map_b, CG_ginn_matrix_b, CG_ginn_node_map_b, DFG_ginn_matrix_b, DFG_ginn_node_map_b) #前向传播
        #print_time("Forward finish")
        #loss = criterion(torch.max(output,-1)[1].float(), label)
        #print('[Debug]',output1,output2,label,output1.shape, output2.shape)
        loss = criterion(output1,output2,label) #loss计算
        total_loss += loss.item()
        total_loss_cnt += 1
        #print_time("Loss cnt finish")
        #print("loss:",loss.item())
        
        if run_type == 'train':
            if(iter_i % 2) == 0:
                print("iter "+str(iter_i)+": total loss ",total_loss/total_loss_cnt)
            mf_model.zero_grad()
            loss.backward() #反向传播
            optimizer.step()
            #print_time("Backward")
        else:
            #结果统计
            euclidean_distance = F.pairwise_distance(output1, output2) #计算距离
            euclidean_distance = euclidean_distance.detach().cpu().numpy().tolist()
            label = label.cpu().numpy().tolist()
            #print(label,euclidean_distance)
            
            st.add_cnt(euclidean_distance,label,run_type) #比较预测距离和真实值
        iter_i += 1
        CFG_As_a, CFG_nodes_a, cg_A_a, DFG_nodes_a, dfg_A_a, DFG_BB_map_a, CG_ginn_matrix_a, CG_ginn_node_map_a, DFG_ginn_matrix_a, DFG_ginn_node_map_a,\
            CFG_As_b, CFG_nodes_b, cg_A_b, DFG_nodes_b, dfg_A_b, DFG_BB_map_b, CG_ginn_matrix_b, CG_ginn_node_map_b, DFG_ginn_matrix_b, DFG_ginn_node_map_b, label = prefetcher.next() #获取下一个batch的数据

    if run_type == 'train':
        print("train epoch_loss:",total_loss/total_loss_cnt)
        scheduler.step(total_loss/total_loss_cnt)


    stop=False
    if run_type != 'train':
        #print("validation----------------------------")
        print(run_type, "epoch_loss,min_loss:",total_loss/total_loss_cnt,cfg.new_config.validation_loss_min)
        #print("---------------------------------------")
        stop = st.print_epoch_reuslt(total_loss/total_loss_cnt,run_type)

    return stop


def model_run(run_type,mf_model,epoch,new_data_loader): #已废弃 
    st = mystatistics.Statistics(cfg.new_config.task,cfg.new_config.subtask)
    total_loss=0
    total_loss_cnt=0
    iter_i=0
    
       
    # debug:another way: this for should in loop before: [for part_i in range(part_amount):]
    for CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map, label in new_data_loader: 
        #print_time("New batch start")
        time_list=[]
        time_list.append(time.time())

        # #padding has moved to dataloader
        # cfg_padding = CFG_As.permute(0,1,3,2)
        # CFG_As = torch.cat((CFG_As,cfg_padding),3)

        # cg_padding = cg_A.permute(0,2,1)
        # cg_A = torch.cat((cg_A,cg_padding),2)

        # dfg_padding = dfg_A.permute(0,2,1)
        # dfg_A = torch.cat((dfg_A,dfg_padding),2)

        # padding = torch.zeros(CFG_nodes.shape[0], CFG_nodes.shape[1], CFG_nodes.shape[2], cfg.new_config.cfg_arg.hidden_dim - CFG_nodes.shape[3]).float()
        # CFG_nodes_x = torch.cat((CFG_nodes, padding), -1)

        # padding = torch.zeros(DFG_nodes.shape[0], DFG_nodes.shape[1], cfg.new_config.dfg_arg.hidden_dim - DFG_nodes.shape[2]).float()
        # DFG_nodes_x = torch.cat((DFG_nodes, padding), -1)

        # time_list.append(time.time())

        # cg_ginn_padding = CG_ginn_matrix.permute(0,1,3,2)
        # CG_ginn_matrix = torch.cat((CG_ginn_matrix,cg_ginn_padding),3)

        # dfg_ginn_padding = DFG_ginn_matrix.permute(0,1,3,2)
        # DFG_ginn_matrix = torch.cat((DFG_ginn_matrix,dfg_ginn_padding),3)

        # #delete no use
        # CFG_nodes_padding = torch.zeros((((cfg.new_config.prog_func_max,cfg.new_config.func_bb_max,cfg.new_config.bb_token_max,cfg.new_config.token_eb_dim-CFG_nodes.size(3)))))
        # CFG_nodes = torch.cat((CFG_nodes,CFG_nodes_padding),3)
        # DFG_nodes_padding = torch.zeros(((cfg.new_config.prog_inst_max,cfg.new_config.inst_token_max,cfg.new_config.token_eb_dim-DFG_nodes.size(2))))
        # DFG_nodes = torch.cat((DFG_nodes,DFG_nodes_padding),2)


        # CFG_As = Variable(CFG_As.cuda()).float()
        # CFG_nodes = Variable(CFG_nodes.cuda()).float()
        # cg_A = Variable(cg_A.cuda()).float()
        # DFG_nodes = Variable(DFG_nodes.cuda()).float()
        # dfg_A = Variable(dfg_A.cuda()).float()
        # DFG_BB_map = Variable(DFG_BB_map.cuda()).float()
        # CG_ginn_matrix = Variable(CG_ginn_matrix.cuda()).float()
        # CG_ginn_node_map = Variable(CG_ginn_node_map.cuda()).float()
        # DFG_ginn_matrix = Variable(DFG_ginn_matrix.cuda()).float()
        # DFG_ginn_node_map = Variable(DFG_ginn_node_map.cuda()).float()
        # label = Variable(label.cuda()).float().to(cfg.new_config.device_label)

        CFG_As = Variable(CFG_As).float()
        CFG_nodes = Variable(CFG_nodes).float()
        #CFG_nodes_x = Variable(CFG_nodes_x).float()
        cg_A = Variable(cg_A).float()
        DFG_nodes = Variable(DFG_nodes).float()
        #DFG_nodes_x = Variable(DFG_nodes_x).float()
        dfg_A = Variable(dfg_A).float()
        DFG_BB_map = Variable(DFG_BB_map).float()
        CG_ginn_matrix = Variable(CG_ginn_matrix).float()
        CG_ginn_node_map = Variable(CG_ginn_node_map).float()
        DFG_ginn_matrix = Variable(DFG_ginn_matrix).float()
        DFG_ginn_node_map = Variable(DFG_ginn_node_map).float()
        label = Variable(label).float().to(cfg.new_config.device_label)

        #print(CFG_As.shape, CFG_nodes.shape, CFG_nodes_x.shape, cg_A.shape, DFG_nodes.shape, DFG_nodes_x.shape, dfg_A.shape)
        #print_time("Start train")
        #output = mf_model(CFG_As, CFG_nodes, CFG_nodes_x, cg_A, DFG_nodes, DFG_nodes_x, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map)
        output = mf_model(CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map)
        
        time_list.append(time.time())
        # print("Compute Graph:")
        # compute_graph = make_dot(output)
        # if cfg.new_config.save_g==False:
        #     cfg.new_config.save_g=True
        #     compute_graph.render("structure_g",view=False,format='pdf')
        label0 = label.argmax(dim=1)
        #print("output:",output.size(),label0.size())
        #print_time("Forward finish")
        #loss = criterion(torch.max(output,-1)[1].float(), label)
        loss = criterion(output,label0.long())
        total_loss += loss.item()
        total_loss_cnt += 1
        #print_time("Loss cnt finish")
        #print("loss:",loss.item())
        if(iter_i % 10) == 0:
            print("iter "+str(iter_i)+": total loss ",total_loss/total_loss_cnt," (lr =",optimizer.param_groups[0]['lr'],")")
        if run_type == 'train':
            # if(iter_i % 10) == 0:
            #     #print('scheduler.step:',total_loss/total_loss_cnt)
            #     scheduler.step(total_loss/total_loss_cnt)
            mf_model.zero_grad()
            loss.backward()
            optimizer.step()
            #print_time("Backward")
        else:
            prediction = F.softmax(output, dim=-1) #output
            prediction = prediction.detach().cpu().numpy().tolist()
            label = label.cpu().numpy().tolist()
            
            st.add_cnt(prediction,label,run_type)
        #print_time("New batch finish")
        time_list.append(time.time())
        time_list_print(time_list)
        iter_i += 1

    if run_type == 'train':
        print("train epoch_loss:",total_loss/total_loss_cnt)
        scheduler.step(total_loss/total_loss_cnt)

    stop=False
    if run_type != 'train':
        #print("validation----------------------------")
        print("validation epoch_loss,min_loss:",total_loss/total_loss_cnt,cfg.new_config.validation_loss_min)
        #print("---------------------------------------")
        stop = st.print_epoch_reuslt(total_loss/total_loss_cnt,run_type)
    return stop


def model_run_data_prefetcher(run_type,mf_model,epoch,new_data_loader):       
    prefetcher = loader.data_next_prefetcher(new_data_loader,cfg.new_config.device1) #预加载器，减少等待数据加载的时间
    st = mystatistics.Statistics(cfg.new_config.task,cfg.new_config.subtask) #初始化统计信息
    total_loss=0
    total_loss_cnt=0
    iter_i=0
       
    #可以优化，不需要列这么多，下面的.float()也可以通过循环处理
    CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map, label = prefetcher.next() #从预加载器拿一个batch的数据
    #for CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map, label in new_data_loader: 
    while label is not None:

        # cg_ginn_padding = CG_ginn_matrix.permute(0,1,3,2)
        # CG_ginn_matrix = torch.cat((CG_ginn_matrix,cg_ginn_padding),3)

        # dfg_ginn_padding = DFG_ginn_matrix.permute(0,1,3,2)
        # DFG_ginn_matrix = torch.cat((DFG_ginn_matrix,dfg_ginn_padding),3)

        # #delete no use
        # CFG_nodes_padding = torch.zeros((((cfg.new_config.prog_func_max,cfg.new_config.func_bb_max,cfg.new_config.bb_token_max,cfg.new_config.token_eb_dim-CFG_nodes.size(3)))))
        # CFG_nodes = torch.cat((CFG_nodes,CFG_nodes_padding),3)
        # DFG_nodes_padding = torch.zeros(((cfg.new_config.prog_inst_max,cfg.new_config.inst_token_max,cfg.new_config.token_eb_dim-DFG_nodes.size(2))))
        # DFG_nodes = torch.cat((DFG_nodes,DFG_nodes_padding),2)


        # CFG_As = Variable(CFG_As.cuda()).float()
        # CFG_nodes = Variable(CFG_nodes.cuda()).float()
        # cg_A = Variable(cg_A.cuda()).float()
        # DFG_nodes = Variable(DFG_nodes.cuda()).float()
        # dfg_A = Variable(dfg_A.cuda()).float()
        # DFG_BB_map = Variable(DFG_BB_map.cuda()).float()
        # CG_ginn_matrix = Variable(CG_ginn_matrix.cuda()).float()
        # CG_ginn_node_map = Variable(CG_ginn_node_map.cuda()).float()
        # DFG_ginn_matrix = Variable(DFG_ginn_matrix.cuda()).float()
        # DFG_ginn_node_map = Variable(DFG_ginn_node_map.cuda()).float()
        # label = Variable(label.cuda()).float().to(cfg.new_config.device_label)

        CFG_As = Variable(CFG_As).float() 
        CFG_nodes = Variable(CFG_nodes).float()
        cg_A = Variable(cg_A).float()
        DFG_nodes = Variable(DFG_nodes).float()
        dfg_A = Variable(dfg_A).float()
        DFG_BB_map = Variable(DFG_BB_map).float()
        CG_ginn_matrix = Variable(CG_ginn_matrix).float()
        CG_ginn_node_map = Variable(CG_ginn_node_map).float()
        DFG_ginn_matrix = Variable(DFG_ginn_matrix).float()
        DFG_ginn_node_map = Variable(DFG_ginn_node_map).float()
        label = Variable(label).float().to(cfg.new_config.device_label)

        # #padding has moved to dataloader
        # cfg_padding = CFG_As.permute(0,1,3,2)
        # CFG_As = torch.cat((CFG_As,cfg_padding),3)

        # cg_padding = cg_A.permute(0,2,1)
        # cg_A = torch.cat((cg_A,cg_padding),2)

        # dfg_padding = dfg_A.permute(0,2,1)
        # dfg_A = torch.cat((dfg_A,dfg_padding),2)

        #with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
        output = mf_model(CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map) #前向传播
        #print(prof.table())
        #prof.export_chrome_trace('./model_profile.json')
        #print(mf_model)
        #ipdb.set_trace()
        # print("Compute Graph:")
        # compute_graph = make_dot(output)
        # if cfg.new_config.save_g==False:
        #     cfg.new_config.save_g=True
        #     compute_graph.render("structure_g",view=False,format='pdf')
        label0 = label.argmax(dim=1)
        #print_time("Forward finish")
        #loss = criterion(torch.max(output,-1)[1].float(), label)
        #print('[Debug] Out:\n',output,label0)
        loss = criterion(output,label0.long()) #计算loss
        total_loss += loss.item()
        total_loss_cnt += 1
        #print("loss:",loss.item())

        if run_type == 'train':
            if(iter_i % 2) == 0:
                print("iter "+str(iter_i)+": total loss ",total_loss/total_loss_cnt)
            # if(iter_i % 10) == 0:
            #     scheduler.step(total_loss/total_loss_cnt)
            mf_model.zero_grad()
            print_memory_occupy('before loss_backward')
            loss.backward() #反向传播

            # for name, parms in mf_model.named_parameters():	
            #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
            #     ' -->grad_value:',parms.grad)

            optimizer.step()
        else:
            #结果统计
            prediction = F.softmax(output, dim=-1) #output
            prediction = prediction.detach().cpu().numpy().tolist()
            label = label.cpu().numpy().tolist()
            
            st.add_cnt(prediction,label,run_type) #比较预测值和真实值
        iter_i += 1
        #detach_pre_tensor([CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map, label])
        CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map, label = prefetcher.next() #获取下一个batch
        #print_time("Next batch data .oad finish")

    if run_type == 'train':
        print("Epoch",i,":","train epoch_loss:",total_loss/total_loss_cnt)
        scheduler.step(total_loss/total_loss_cnt) #根据loss动态修改lr

    stop=False
    if run_type != 'train':
        #print("validation----------------------------")
        print("Epoch",i,":",run_type,"epoch_loss,min_loss:",total_loss/total_loss_cnt,cfg.new_config.validation_loss_min)
        #print("---------------------------------------")
        stop = st.print_epoch_reuslt(total_loss/total_loss_cnt,run_type) #输出这个epoch的valdidation和test结果
    
    return stop


# def create_sub_data_loader(index, data_type): 
#     print("Load part data start("+str(index)+"):",time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time())))
#     sample_list = program.load_pre_data(index,cfg.new_config.max_per_load,data_type)
#     print("Load part data finish("+str(index)+"):",time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time())))
#     new_data_set = program.to_data_set(sample_list)
    
#     kwargs = {'num_workers': 8, 'pin_memory': False} if cfg.new_config.cuda else {}
#     new_sub_data_loader = torch.utils.data.DataLoader(
#                         loader.MyDataset(new_data_set.CFG_As,new_data_set.CFG_nodes,new_data_set.cg_A,new_data_set.DFG_nodes,new_data_set.dfg_A,new_data_set.DFG_BB_map,new_data_set.CG_ginn_matrix,new_data_set.CG_ginn_node_map,new_data_set.DFG_ginn_matrix,new_data_set.DFG_ginn_node_map,new_data_set.label), batch_size=cfg.new_config.batch, **kwargs)
#     print("Create part data loader finish (size ="+str(len(new_data_set.label))+"):",time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time())))
#     return new_sub_data_loader

# def get_sample_amount(set_type):
#     sample_amount=os.popen('ls '+cfg.new_config.pre_data_path+set_type+' | wc -l')
#     sample_amount_str = sample_amount.readlines()[0][:-1]
#     sample_amount = int(sample_amount_str)
#     return sample_amount

# def get_pre_data_size():
#     cfg.new_config.pre_sample_max_train = get_sample_amount('train')
#     cfg.new_config.pre_sample_max_validation = get_sample_amount('validation')
#     cfg.new_config.pre_sample_max_test = get_sample_amount('test')
#     print("Data size:",cfg.new_config.pre_sample_max_train,cfg.new_config.pre_sample_max_validation,cfg.new_config.pre_sample_max_test)

def weight_init(m): #修改模型参数的初始化
    if isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight)
        #nn.init.constant_(m.bias,0)
    elif isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
    elif isinstance(m,nn.BatchNorm2d):
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)

def debug_init():
    #在运行binary相似性检测之前，需要先把所有的样例组成pair，所以先取消注释debug_init函数，执行loader.init_print_list -> pair_sample.init_print_list来完成pair的组对
    #pair_sample.init_print_list将组队好的结果(train/test/validation每个集合一个list，list中为样例的目录，这些list被存储到文件中，train.py执行相似性任务时会直接读取这个文件)
    loader.init_print_list()

if __name__ == '__main__':
    # 生成binary相似性检测任务的数据集，生成数据时注销这两行；生成结束后注释这两行
    # debug_init()
    # exit()

    print("Work start:",time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time())))
    opt = parser.parse_args()
    config=cfg.CONFIG(opt)
    cfg.new_config = config

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.new_config.gpu #设置使用的GPU
    model_name = cfg.new_config.task+'_'+cfg.new_config.model_type+'_'+cfg.new_config.model_tag+'_'+time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time()))
    output_to='file'
    if cfg.new_config.debug:
        output_to='std'

    log_file = cfg.log_file(output_to,'log/run_'+time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time()))) #设置输出结果存储路径
    cfg.new_config.print_c()
    
    f_dir = cfg.new_config.pre_data_path +'/'
    compile_class_tag_list = cfg.new_config.compiler_tag_list.split(',')
    data_list = []
    if cfg.new_config.task=='deadstore': #路径需要修改为自己的存储方式，之前的实验中将deadstore label为0的放在了0/文件夹下，label为1的放在了1/文件夹下，如果没有分开放，则不需要分两个路径加载
        for compile_class_tag in compile_class_tag_list:
            print('DATA PATH',f_dir+"0/sample_"+compile_class_tag+"*.pkl")
            data_tmp1 = glob.glob(f_dir+"0/sample_"+compile_class_tag+"*.pkl")
            data_list = data_list + data_tmp1
            print('DATA PATH',f_dir+"1/sample_"+compile_class_tag+"*.pkl")
            data_tmp2 = glob.glob(f_dir+"1/sample_"+compile_class_tag+"*.pkl")
            data_list = data_list + data_tmp2
            print('DATALIST COMPILE_CLASS:',compile_class_tag,' LENGTH (',len(data_tmp1),'/',len(data_tmp2),')')
    else:
        for compile_class_tag in compile_class_tag_list:
            print('DATA PATH',f_dir+"sample_"+compile_class_tag+"*.pkl")
            data_tmp = glob.glob(f_dir+"sample_"+compile_class_tag+"*.pkl")
            data_list = data_list + data_tmp
            print('DATALIST COMPILE_CLASS:',compile_class_tag,' LENGTH',len(data_tmp))
        
    random.shuffle(data_list) #把所有数据随机排序，保证后续使用时是随机的
    
    kwargs = {'num_workers': 8, 'pin_memory': False} if cfg.new_config.cuda else {}

    #定义数据加载器
    if cfg.new_config.task=='binaryClassify' and cfg.new_config.subtask=='binaryPair':
        #torch.utils.data.DataLoader #original
        #loader.PrefetchDataLoader #speedup
        validation_data_loader = loader.PrefetchDataLoader(loader.NewDataset_bert_pair('validation'), batch_size=cfg.new_config.batch, **kwargs)
        test_data_loader = loader.PrefetchDataLoader(loader.NewDataset_bert_pair('test'), batch_size=cfg.new_config.batch, **kwargs)
        train_data_loader = loader.PrefetchDataLoader(loader.NewDataset_bert_pair('train'), batch_size=cfg.new_config.batch, shuffle=True, **kwargs)
    if cfg.new_config.task=='deadstore':
        train_data_loader = loader.PrefetchDataLoader(loader.DeadDataset('train',data_list), batch_size=cfg.new_config.batch, shuffle=True, **kwargs)
        validation_data_loader = loader.PrefetchDataLoader(loader.DeadDataset('validation',data_list), batch_size=cfg.new_config.batch, **kwargs)
        test_data_loader = loader.PrefetchDataLoader(loader.DeadDataset('test',data_list), batch_size=cfg.new_config.batch, **kwargs)
    else:
        train_data_loader = loader.PrefetchDataLoader(loader.NewDataset_bert('train',data_list), batch_size=cfg.new_config.batch, shuffle=True, **kwargs)
        validation_data_loader = loader.PrefetchDataLoader(loader.NewDataset_bert('validation',data_list), batch_size=cfg.new_config.batch, **kwargs)
        test_data_loader = loader.PrefetchDataLoader(loader.NewDataset_bert('test',data_list), batch_size=cfg.new_config.batch, **kwargs)
        

    # train_data_loader = loader.data_prefetcher(train_data_loader,cfg.new_config.device1)
    # validation_data_loader = loader.data_prefetcher(validation_data_loader,cfg.new_config.device1)
    # test_data_loader = loader.data_prefetcher(test_data_loader,cfg.new_config.device1)

    
    
    if cfg.new_config.task=='binaryClassify' and cfg.new_config.subtask=='binaryPair':
        criterion = model.ContrastiveLoss().to(cfg.new_config.device_label) #siamese框架需要这个criterion
    else:
        criterion = nn.CrossEntropyLoss().to(cfg.new_config.device_label) #其中包含了softmax

    if cfg.new_config.load_test=="null": #根据设置初始化模型
        if cfg.new_config.task=='binaryClassify' and cfg.new_config.subtask=='binaryPair':
            mf_model = model.model_pair(cfg.new_config)
        elif cfg.new_config.model_type=="cfg_bert":
            mf_model = model.cfg_bert(cfg.new_config)
        elif cfg.new_config.model_type=="cg_bert":
            mf_model = model.cg_bert(cfg.new_config)
        elif cfg.new_config.model_type=="dfg_bert":
            mf_model = model.dfg_bert(cfg.new_config)
        elif cfg.new_config.model_type=="cg_only":
            mf_model = model.cg_only(cfg.new_config)
        elif cfg.new_config.model_type=="no_cg":
            mf_model = model.no_cg(cfg.new_config)
        elif cfg.new_config.model_type=="dead_cg":
            mf_model = model.dead_cg(cfg.new_config)
        elif cfg.new_config.model_type=="all":
            mf_model = model.complete(cfg.new_config)
        mf_model = mf_model.float()
    else: #从已有的模型加载模型
        mf_model = torch.load('model_save/tmp_model/'+cfg.new_config.load_test,map_location=cfg.new_config.device1)
        # mf_model.to(cfg.new_config.device1)
        print("model to device finish:",cfg.new_config.device1)
    
    mf_model.to(cfg.new_config.device1)

    lr = cfg.new_config.lr #* (0.1 ** (epoch // 15))
    if cfg.new_config.weight_decay < 0:
        optimizer = torch.optim.Adam(mf_model.parameters(), lr=lr)  
    else:
        optimizer = torch.optim.Adam(mf_model.parameters(), lr=lr, weight_decay=cfg.new_config.weight_decay)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=1, min_lr=0, eps=1e-08) #动态lr

    # if cfg.new_config.lossfunc == "focal":
    #     print("lossfunc:focal")
    #     criterion = mymodel.BCEFocalLoss().to(cfg.new_config.device_label)
    # else:
    #     criterion = torch.nn.BCELoss().to(cfg.new_config.device_label)
    #criterion = nn.BCEWithLogitsLoss().to(cfg.new_config.device_label)
    
    #mf_model.apply(weight_init)


    print("Train start:",time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time())))
    for i in range(1,cfg.new_config.epoch+1): #开始训练
        print("Epoch:",i," train( lr =",optimizer.param_groups[0]['lr'],", time =", time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time())),")")
        mf_model.train() #设置为训练模型
        
        if cfg.new_config.task=='binaryClassify' and cfg.new_config.subtask=='binaryPair':
            stop = model_run_pair_prefetcher('train',mf_model,i,train_data_loader) #实际迭代
        else:
            stop = model_run_data_prefetcher('train',mf_model,i,train_data_loader) #实际迭代
        # scheduler.step()

        #torch.cuda.empty_cache()
        print("Epoch:",i," validation")
        mf_model.eval() #设置模型为非训练模式，节省内存开销
        with torch.no_grad():
            if cfg.new_config.task=='binaryClassify' and cfg.new_config.subtask=='binaryPair':
                stop_validation = model_run_pair_prefetcher('validation',mf_model,i,validation_data_loader) #validation
            else:
                stop_validation = model_run_data_prefetcher('validation',mf_model,i,validation_data_loader) #validation
        #torch.cuda.empty_cache()
        print("Epoch:",i," test")
        with torch.no_grad():
            if cfg.new_config.task=='binaryClassify' and cfg.new_config.subtask=='binaryPair':
                stop = model_run_pair_prefetcher('test',mf_model,1,test_data_loader) #test
            else:   
                stop = model_run_data_prefetcher('test',mf_model,1,test_data_loader) #test

        #torch.cuda.empty_cache()
        if i <10 :
            stop_validation=False #至少训练10个epoch
        if stop_validation:
            break
        if cfg.new_config.save_model:
            torch.save(mf_model, 'model_save/tmp_model/'+model_name+'_Epoch_'+str(i)+'.pkl') #保存这个epoch训练后的模型
    print("Train finish:",time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time())))

    #如果是加载了模型，不训练只测试（epoch被设置为0）
    if cfg.new_config.epoch==0:
        print("Test for a saved model.")
        with torch.no_grad():
            if cfg.new_config.task=='binaryClassify' and cfg.new_config.subtask=='binaryPair':
                stop = model_run_pair_prefetcher('test',mf_model,1,test_data_loader)
            else:   
                stop = model_run_data_prefetcher('test',mf_model,1,test_data_loader)

    #保存最终的模型
    if cfg.new_config.save_model:
        torch.save(mf_model, 'model_save/'+model_name+'.pkl')

    log_file.close()

