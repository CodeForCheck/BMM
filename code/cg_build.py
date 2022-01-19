#! /usr/bin/env python
import sys  
import networkx as nx
import numpy as np
import angr
import argparse
import os
from collections import defaultdict
import scipy.sparse as sp
import pickle
import json


parser = argparse.ArgumentParser()
parser.add_argument('--target_program', type=str, default='null', help='target_program type(poj/spec)')
parser.add_argument('--arch_t', type=str, default='x86', help='type of arch')
parser.add_argument('--output', type=str, default='./', help='output path')
parser.add_argument('--input', type=str, default='Sourcecode', help='input path')
parser.add_argument('--comp_t', type=str, default='gcc', help='type of compiler')
parser.add_argument('--opti_t', type=str, default='O2', help='type of optimization')
parser.add_argument('--pro_class', type=str, default='0', help='program class')
parser.add_argument('--filename', type=str, default='0', help='program filename')
#python cg_build.py --output=poj_data --comp_t=gcc --opti_t=O3 --pro_class=1 --pro_id=1020

child_dic = defaultdict(int)
father_dic = defaultdict(int)
func_list=[]


# def create_subgraph(G,sub_G,start_node,hop):
#     if hop == 0:
#         return
#     for n in G.successors(start_node):
#         if (n not in sub_G) and (n in func_list): # and (sub_G.number_of_nodes()<256):
#             #if n in func_memory_dic:
#             #    if (func_memory_dic[n]&node_set):
#             sub_G.add_node(n)
#             sub_G.add_edge(start_node,n)
#             #child_list.append(n)
#             if (father_dic[n]+child_dic[n])<20:  #gcc:100:
#                 create_subgraph(G,sub_G,n,hop-1)
#     for n in G.predecessors(start_node):
#         if (n not in sub_G) and (n in func_list): # and (sub_G.number_of_nodes()<256):
#             #if n in func_memory_dic:
#             #    if (func_memory_dic[n]&node_set):
#             sub_G.add_node(n)
#             sub_G.add_edge(n,start_node)
#             #father_list.append(n)
#             if (father_dic[n]+child_dic[n])<20:  #gcc:100:     
#                 create_subgraph(G,sub_G,n,hop-1)
#     '''
#     for n in child_list:
#         create_subgraph(G,sub_G,n,hop-1)
#     for n in father_list:
#         create_subgraph(G,sub_G,n,hop-1)
#     '''


def analyze(b, addr, name=None):
    cfg = b.analyses.CFGFast()  #先构建cfg
    cg = cfg.functions.callgraph #通过cfg得到cg
    #print(cg.number_of_nodes(),len(func_list))
    A=np.array(nx.adjacency_matrix(cg).todense()) #得到邻接矩阵
    AS = sp.lil_matrix(A) #turn to sparse matrix of A_block to save space
    np.save(f_adj,AS) 
    print(cg.number_of_nodes(),file=f_node)
    for n in cg.nodes(): #输出cg的每个节点
        print(n,file=f_node)
    
    # for node in cg.nodes():
    #     child_n=0
    #     father_n=0
    #     for n in cg.successors(node):
    #         child_n += 1
    #     for n in cg.predecessors(node):
    #         father_n += 1
    #     child_dic[node]=child_n
    #     father_dic[node]=father_n

    # for node in cg.nodes():
    #     sub_G = nx.DiGraph()
    #     sub_G.add_node(node)
    #     if (node in func_list):
    #         #node_set={}
    #         #if node in func_memory_dic:
    #         #    node_set = func_memory_dic[node]
    #         create_subgraph(cg, sub_G,node,5)
    #         sub_A=np.array(nx.adjacency_matrix(sub_G).todense())
    #         #print(sub_A)
    #         #print(node,sub_G.number_of_nodes())
    #         np.savetxt(f_adj,sub_A)
    #         print(node,sub_G.number_of_nodes(),file=f_node)
    #         for n in sub_G.nodes():
    #             print(n,end=" ",file=f_node)
    #         print("\n",end="",file=f_node)

    


# def create_dic(func_dic,f):
#     for line in f:
#         items = line.split()
#         func_dic[items[0]].add(str(items[1]))


if __name__ == "__main__":
    arg = parser.parse_args()
    pro_class = arg.pro_class
    filename = arg.filename
    comp_t = arg.comp_t
    opti_t = arg.opti_t
    output_path = arg.output
    input_path = arg.input
    arch_t = arg.arch_t
    target_program = arg.target_program
    if target_program=="poj": #构造路径，按自己的存储方式修改
        in_path='/home/angr/workspace/POJ/'+input_path+'/'+pro_class+'/'+filename+'-'+comp_t+'-'+opti_t
        out_name = output_path+'/'+comp_t+'/'+opti_t+'/'+pro_class+'/'+filename
    elif target_program=="spec":
        in_path='./'+arch_t+'_'+comp_t+'_'+opti_t+'_test_benchmark/'+filename
        if opti_t=="o2":
            opti_t_out="O2"
        elif opti_t=="o3":
            opti_t_out="O3"
        else:
            print("Error opti_t.")
        out_name = output_path+'/'+arch_t+'/'+comp_t+'/'+opti_t_out+'/'+filename
    else:
        in_path="./"
        out_name = "./"
    
    print(out_name)

     
    f_node_name = out_name+"_cg_arg"
    f_node = open(f_node_name,'w+')
    f_adj_name = out_name+"_cg_adj"
    f_adj = open(f_adj_name,'wb+')
    proj = angr.Project(in_path, load_options={'auto_load_libs':False}) #加载binary到proj，angr用proj对象来做各类解析

    
    # #f_label_name = "cfg_label/"+name+"_label"
    # f_label_name = ptype+"_result/label_"+name
    # arg_f = open(f_label_name,'r')
    # arg_line = arg_f.readline()
    # while arg_line:
    #     addr,bb_cnt,cfg_label = map(int,arg_line.split())
    #     if addr not in func_list:
    #         func_list.append(addr)
    #     arg_line = arg_f.readline()
    #     arg_line = arg_f.readline()
    #     arg_line = arg_f.readline()
    
    main = proj.loader.main_object.get_symbol("main") #main函数起始地址
    analyze(proj, main.rebased_addr) #生成cg

