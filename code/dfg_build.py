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
#python dfg_build.py --output=poj_data --comp_t=gcc --opti_t=O2 --pro_class=1 --pro_id=1020

child_dic = defaultdict(int)
father_dic = defaultdict(int)
func_list=[]
pro_id='0'


def analyze(b, addr, name=None):
    cfg = b.analyses.CFGEmulated(resolve_indirect_jumps=False,context_sensitivity_level=2, keep_state=True,state_add_options=angr.sim_options.refs)  #先得到cfg
    ddg = b.analyses.DDG(cfg,start=cfg.functions['main'].addr) #通过cfg得到dfg
    A=np.array(nx.adjacency_matrix(ddg.graph).todense()) #得到邻接矩阵
    AS = sp.lil_matrix(A) #turn to sparse matrix of A_block to save space
    np.save(f_adj,AS)
    print(ddg.graph.number_of_nodes(),file=f_node)
    for n in ddg.graph.nodes():
        print(n,file=f_node) #输出dfg的每个节点


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
    #构造输入输出地址
    if target_program=="poj": 
        in_path='/home/angr/workspace/POJ/'+input_path+'/'+pro_class+'/'+filename+'-'+comp_t+'-'+opti_t
        out_name = output_path+'/'+comp_t+'/'+opti_t+'/'+pro_class+'/'+filename #same path; different name with cfg
    elif target_program=="spec":
        in_path='./'+arch_t+'_'+comp_t+'_'+opti_t+'_test_benchmark/'+filename
        if opti_t=="o2":
            opti_t_out="O2"
        elif opti_t=="o3":
            opti_t_out="O3"
        else:
            print("Error opti_t.")
        out_name = output_path+'/'+arch_t+'/'+comp_t+'/'+opti_t_out+'/'+filename #same path; different name with cfg
    else:
        in_path="./"
        out_name = "./" 
    print(out_name)

    f_node_name = out_name+"_dfg_arg"
    f_node = open(f_node_name,'w+')
    f_adj_name = out_name+"_dfg_adj"
    f_adj = open(f_adj_name,'wb+')
    proj = angr.Project(in_path, load_options={'auto_load_libs':False},default_analysis_mode='symbolic') #加载binary到proj，angr用proj对象来做各类解析

       
    main = proj.loader.main_object.get_symbol("main") #main函数起始地址
    analyze(proj, main.rebased_addr) #生成dfg

