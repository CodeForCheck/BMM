import sys
import numpy as np
import torch
from collections import defaultdict
new_config = None

class GraphConfig():
    def __init__(self,init_dim,hidden_dim,out_dim,steps,node_max,task):
        self.init_dim = init_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.steps = steps
        self.node_max = node_max
        self.task_type = task
    def printg(self):
        print("init_dim =",self.init_dim)
        print("hidden_dim =",self.hidden_dim)
        print("out_dim =",self.out_dim)
        print("steps =",self.steps)
        print("node_max =",self.node_max)
        print("task_type =",self.task_type)

class CONFIG():
    def __init__(self,opt):
        self.check_arg(opt)
        self.cuda = True
        self.debug = opt.debug
        self.pre_data_path = opt.pre_data_path
        self.gpu = opt.gpu
        self.device1 = torch.device("cuda:0")
        self.device2 = torch.device("cuda:1")
        self.device3 = torch.device("cuda:2")
        self.device4 = torch.device("cuda:3")
        self.device_label = self.device1
        self.epoch = opt.epoch
        self.batch = opt.batch
        self.lr = opt.lr
        self.weight_decay = opt.weight_decay
        self.pre_sample_max_train = 128 #14490
        self.pre_sample_max_test = 64 #2000
        self.pre_sample_max_validation = 64 #2000
        self.max_per_load = self.batch * 128
        self.stop_epoch = 0
        self.lossfunc= opt.lossfunc
        self.drop_rate = opt.drop_rate
        self.copy_rate = opt.copy_rate
        self.validation_loss_min=1000
        self.valid_loss_stop_down=0
        self.model_type = opt.model_type
        self.token_eb_model = opt.token_eb_model
        self.model_tag = opt.model_tag
        self.save_model = opt.save_model
        self.accuracy_max = 0
        self.accuracy_stop_raise = 0
        self.task = opt.task
        self.subtask = opt.subtask
        self.compiler_tag_list = opt.compiler_tag_list
        self.load_test = opt.load_test
        self.train_rate = 0.7
        self.validation_rate = 0.15
        self.test_rate = 0.15
        
        self.token_eb_dim = opt.token_eb_dim
        self.new_token_eb_dim = opt.new_token_eb_dim 
        self.inst_eb_dim = opt.inst_eb_dim
        self.bb_eb_dim = opt.bb_eb_dim
        self.func_eb_dim = opt.func_eb_dim
        self.inst_token_max = opt.inst_token_max #16 
        self.bb_token_max = opt.bb_token_max #128 
        self.func_bb_max = opt.func_bb_max #32
        self.prog_func_max = opt.prog_func_max #16
        self.prog_inst_max = opt.prog_inst_max #4096
        self.ginn_max_level = opt.ginn_max_level #max:4
        self.ginn_window = 8

        self.func_class = 104
        self.bug_class = self.prog_func_max * self.func_bb_max
        #self.binary_class = 16 #TODO:change class amount
        self.similarity_class = 128

        #all graph arg
        graph_node_max = 256
        #arg for cfg GGNN
        self.cfg_arg = GraphConfig(opt.cfg_init_dim,opt.cfg_hidden_dim,opt.cfg_out,opt.cfg_steps,self.func_bb_max,opt.task)
        # self.cfg_init_dim=opt.cfg_init_dim
        # self.cfg_hidden_dim=opt.cfg_hidden_dim
        # self.cfg_out=opt.cfg_out
        # self.cfg_steps=opt.cfg_steps
        # self.cfg_node_max = 256
        
        #arg for cg GGNN
        self.cg_arg = GraphConfig(opt.cg_init_dim,opt.cg_hidden_dim,opt.cg_out,opt.cg_steps,self.prog_func_max,opt.task)
        # self.cg_init_dim = opt.cg_init_dim
        # self.cg_node_max = 256
        # self.cg_edge_type = 1
        # self.cg_out = opt.cg_out #for task 18
        # self.cg_task_id = 18
        # self.cg_hidden_dim=opt.cg_hidden_dim
        # self.cg_steps = opt.cg_steps
        
        #arg for dfg GGNN
        self.dfg_arg = GraphConfig(opt.dfg_init_dim,opt.dfg_hidden_dim,opt.dfg_out,opt.dfg_steps,self.prog_inst_max,opt.task)
        # self.dfg_init_dim = opt.dfg_init_dim
        # self.dfg_node_max = 256
        # self.dfg_edge_type = 1
        # self.dfg_out = opt.dfg_out #for task 18
        # self.dfg_task_id = 18
        # self.dfg_hidden_dim=opt.dfg_hidden_dim
        # self.dfg_steps = opt.dfg_steps

        #arg for MLP
        if self.model_type=="cfg_bert":
            self.fc_input_size=self.cfg_arg.out_dim
        elif self.model_type=="cg_bert":
            self.fc_input_size=self.cg_arg.out_dim
        elif self.model_type=="dfg_bert":
            self.fc_input_size=self.dfg_arg.out_dim
        elif self.model_type=="cg_only":
            self.fc_input_size=self.cg_arg.out_dim
        elif self.model_type=="no_cg":
            self.fc_input_size=self.cfg_arg.out_dim + self.dfg_arg.out_dim
        elif self.model_type=="dead_cg":
            self.fc_input_size=self.cg_arg.out_dim
        elif self.model_type=="dead_cfg":
            self.fc_input_size=self.cfg_arg.out_dim
        elif self.model_type=="all":
            self.fc_input_size=self.cg_arg.out_dim + self.dfg_arg.out_dim
        
        self.common_size=1

        self.compiler_list=self.compiler_tag_list.split(',')
        self.compiler_dic = {}
        index = 0
        for t in self.compiler_list:
            self.compiler_dic[int(t)] = index
            index += 1
        self.binary_class = len(self.compiler_dic)


        #others:
        self.save_g=False

    def check_arg(self, opt):
        #TODO：添加才参数检查
        if opt.cfg_init_dim > opt.cfg_hidden_dim:
            print('[Error] cfg_init_dim > cfg_hidden_dim', opt.cfg_init_dim, opt.cfg_hidden_dim)
        if opt.cg_init_dim > opt.cg_hidden_dim:
            print('[Error] cg_init_dim > cg_hidden_dim', opt.cg_init_dim, opt.cg_hidden_dim)
        if opt.dfg_init_dim > opt.dfg_hidden_dim:
            print('[Error] dfg_init_dim > dfg_hidden_dim', opt.dfg_init_dim, opt.dfg_hidden_dim)
        # if opt.cg_init_dim < opt.cfg_out:
        #     print('[Error] cfg_out > cg_init_dim',opt.cfg_out, opt.cg_init_dim)
        if opt.cg_init_dim != opt.cfg_out:
            print('[Error] cfg_out != cg_init_dim',opt.cfg_out, opt.cg_init_dim)
            


    def reset_model(self): #已废弃
        self.stop_epoch = 0
        self.maxRP = 0
        self.validation_loss_min=1000
        self.valid_loss_stop_down=0
        self.accuracy_max = 0
        self.accuracy_stop_raise = 0
 
        #arg for MLP
        if self.model_type=="cfg_bert":
            self.fc_input_size=self.cfg_out
        elif self.model_type=="cg_bert":
            self.fc_input_size=self.cg_out
        elif self.model_type=="dfg_bert":
            self.fc_input_size=self.dfg_out
        elif self.model_type=="cg_only":
            self.fc_input_size=self.cg_out
        elif self.model_type=="no_cg":
            self.fc_input_size=self.dfg_out+self.cfg_out
        elif self.model_type=="dead_cg":
            self.fc_input_size=self.cg_out
        elif self.model_type=="dead_cfg":
            self.fc_input_size=self.cfg_out
        elif self.model_type=="all":
            self.fc_input_size=self.cfg_out + self.cg_out + self.dfg_out
        

    def print_c(self): #打印所有参数
        print("debug =",self.debug)
        print("pre_data_path =",self.pre_data_path)
        print("pre_sample_max_train =",self.pre_sample_max_train)
        print("pre_sample_max_test =",self.pre_sample_max_test)
        print("pre_sample_max_validation =",self.pre_sample_max_validation)
        print("max_per_load =",self.max_per_load)
        print("gpu =",self.gpu)
        print("device1 =",self.device1)
        print("device2 =",self.device2)
        print("device3 =",self.device3)
        print("device4 =",self.device4)
        print("epoch =",self.epoch)
        print("batch =",self.batch)
        print("lr =",self.lr)
        print("weight_decay =",self.weight_decay)
        print("stop_epoch =",self.stop_epoch)
        print("lossfunc =",self.lossfunc)
        print("drop_rate =",self.drop_rate)
        print("copy_rate =",self.copy_rate)
        print("model_type =",self.model_type)
        print("token_eb_model =",self.token_eb_model)
        print("model_tag =",self.model_tag)
        print("save_model =",self.save_model)
        print("accuracy_max =",self.accuracy_max)
        print("accuracy_stop_raise =",self.accuracy_stop_raise)
        print("task =",self.task)
        print("subtask",self.subtask)
        print("compiler_tag_list",self.compiler_tag_list)
        print("load_test =",self.load_test)
        print("train_rate =",self.train_rate)
        print("validation_rate =",self.validation_rate)
        print("test_rate =",self.test_rate)
        
        print("token_eb_dim =",self.token_eb_dim) 
        print("new_token_eb_dim =",self.new_token_eb_dim) 
        print("inst_eb_dim =",self.inst_eb_dim)
        print("bb_eb_dim =",self.bb_eb_dim)
        print("func_eb_dim =",self.func_eb_dim)
        print("inst_token_max =",self.inst_token_max)
        print("bb_token_max =",self.bb_token_max)
        print("func_bb_max =",self.func_bb_max)
        print("prog_func_max =",self.prog_func_max)
        print("prog_inst_max =",self.prog_inst_max)
        print("ginn_max_level =",self.ginn_max_level)
        print("ginn_window =",self.ginn_window)

        print("func_class =",self.func_class)
        print("bug_class =",self.bug_class)
        print("binary_class =",self.binary_class)
        print("similarity_class =",self.similarity_class)

        print("fc_input_size =",self.fc_input_size)
        print("cfg arg:")
        self.cfg_arg.printg()
        print("cg arg:")
        self.cg_arg.printg()
        print("dfg arg:")
        self.dfg_arg.printg()



class log_file():
    def __init__(self,out_type,path):
        self.type = out_type
        self.log_path = path
        if out_type == 'file': #如果要输出到文件
            self.saveout = sys.stdout
            self.log_f = open(self.log_path,'w')
            sys.stdout = self.log_f

            self.saveerr = sys.stderr
            self.err_f = open(self.log_path+'_err','w')
            sys.stderr = self.err_f
        
            
    def new_log(self,list):
        for token in list:
            log_f.write(token,end=" ")
        log_f.write(end="")

    def close(self): #关闭日志文件
        if self.type == 'file':
            sys.stdout = self.saveout
            sys.stderr = self.saveerr
            self.log_f.close()
            self.err_f.close()

def get_size(obj): #未使用
    if type(obj)==list:
        return "list["+str(len(obj))+"]"
    elif type(obj)==int:
        return "int("+str(obj)+")"
    elif type(obj)==np.ndarray:
        return "matrix:"+str(obj.shape)
    elif type(obj)==float:
        return "float("+str(obj)+")"
    


