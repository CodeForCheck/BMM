#! /usr/bin/env python
import sys  
import networkx as nx
import numpy as np
import angr
import argparse
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
#python cfg_build.py --output=poj_data --comp_t=gcc --opti_t=O2 --pro_class=1 --pro_id=1020


class CFG:
    def __init__(self, addr, block_cnt, ins_cnt_list, adjm):
        self.addr = addr
        self.block_cnt = block_cnt
        self.ins_cnt_list = np.array(ins_cnt_list)
        self.adjm = adjm

def cfg2dict(std):
    if isinstance(std,CFG):
        return {
            'addr':std.addr,
            'block_cnt':std.block_cnt,
            'ins_cnt_list':std.ins_cnt_list,
            'adjm':std.adjm
        }



def analyze(b, addr, name=None):
    start_state = b.factory.blank_state(addr=addr)
    start_state.stack_push(0x0)
    cfg = b.analyses.CFGFast() #生成cfg
    for addr,func in cfg.kb.functions.items(): #遍历所有函数的cfg
        graph_f = func.transition_graph
        i=0
        block_cnt=0
        node_list=[]
        for node in func.nodes():
            try:
                if node.is_blocknode: #排除其他node，只保留blocknode
                    node_list.append(i)
                    block_cnt = block_cnt+1
            except:
                pass
            i=i+1
        #print("node_list",node_list)
        block_cnt_r=0
        for block in func.blocks:
            block_cnt_r = block_cnt_r+1
        if block_cnt_r != block_cnt:
            continue
        if block_cnt==0:
            continue
        A=np.array(nx.adjacency_matrix(graph_f).todense()) #得到邻接矩阵
        A_block = A[:,node_list] #仅保留有效节点的边
        A_block = A_block[node_list,:]
        A_block = A_block.astype(np.int16)
        AS = sp.lil_matrix(A_block) #turn to sparse matrix of A_block to save space

        #print to 1 file
        # ins_cnt_list=[]
        # for block in func.blocks:
        #     if block.size>0:
        #         ins_cnt_list.append(block.instructions)
        #         block.pp()
        #     else:
        #         ins_cnt_list.append(0)
        # cfg_sample = CFG(addr,block_cnt,ins_cnt_list)
        # cfg_sample = CFG(addr,block_cnt)
        # #pickle.dump(cfg_sample,f_arg)
        # json.dump(cfg_sample,f_arg,default=cfg2dict)



        # #print to 3 file (DAC)
        print(addr,block_cnt,file=f_arg)
        #np.savetxt(f_adj,A_block.astype(int))
        #sp.save_npz(f_adj, AS)
        #输出方式不好，推荐使用spec_dfg_build.py的方式，但为了衔接，还是沿用了这种方式
        np.save(f_adj,AS)
        for block in func.blocks:
            if block.size>0:
                print(block.instructions,end=" ",file=f_arg)
                block.pp()
            else:
                print(0,end=" ",file=f_arg)
        print("\n",file=f_arg)
           
if __name__ == "__main__":
    arg = parser.parse_args()
    pro_class = arg.pro_class
    filename = arg.filename
    comp_t = arg.comp_t
    opti_t = arg.opti_t
    output_path = arg.output
    arch_t = arg.arch_t
    target_program = arg.target_program
    input_path = arg.input
    if target_program=="poj": #构造路径
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

    save_stdout=sys.stdout
    f_bb_name = out_name+"_bb"
    f_adj_name = out_name+"_adj"
    f_arg_name = out_name+"_arg"
    f_bb = open(f_bb_name,'w+')
    f_adj = open(f_adj_name,'wb+')
    f_arg = open(f_arg_name,'w+')
    sys.stdout=f_bb
    proj = angr.Project(in_path, load_options={'auto_load_libs':False}) #加载binary到proj，angr用proj对象来做各类解析
    #debug
    #cfg0 = proj.analyses.CFGFast()
    #print("debug: cfg0 finish\n")

    #proj = angr.Project("test", load_options={'auto_load_libs':False})
    main = proj.loader.main_object.get_symbol("main") #main函数起始地址
    analyze(proj, main.rebased_addr) #生成cfg
    f_adj.close()
    f_arg.close()
    f_bb.close()
    sys.stdout=save_stdout


    # #debug
    # f_adj_o = open(f_adj_name,'rb')
    # f_arg_o = open(f_arg_name,'r')
    # arg_line = f_arg_o.readline()
    # while(arg_line):
    #     addr,bb_cnt = map(int,arg_line.split())
    #     arg_line = f_arg_o.readline()
    #     ins_cnt = list(map(int,arg_line.split()))
    #     arg_line = f_arg_o.readline()
    #     arg_line = f_arg_o.readline()
    #     print(addr,bb_cnt)
    #     AS = np.load(f_adj_o,allow_pickle=True)[()]
    #     A = AS.toarray()
    #     print(A)













    


