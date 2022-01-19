#copy from model onedevice.py
import torch
import torch.nn as nn
import cfg
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch import cuda
import time

def print_memory_occupy(runtag):
    return
    print("[Memory]",runtag, cuda.memory_allocated(cfg.new_config.device1)/1024**2)
    #print("  [Time]",time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time())))

def time_list_print(time_list):
    return
    if len(time_list)<2:
        return
    print('[Time MODEL]')
    for i in range(0,len(time_list)-1):
        print(round(time_list[i+1] - time_list[i],4),end=' ')
    print('')


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    def forward(self, _input, target):
        pt = _input
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
            (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

class LSTM_CFG(nn.Module): #废弃
    def __init__(self, in_dim):
        super(LSTM_CFG, self).__init__()
        self.lstm = nn.LSTM(input_size=in_dim,hidden_size=2*in_dim,batch_first=True,bidirectional=True)
        self.fc_out = nn.Linear(4*in_dim, in_dim)

    def forward(self, x):
        # 1->3 2->4 3->5
        #[batch,config.prog_func_max,config.func_bb_max,config.bb_token_max,config.inst_eb_dim]
        print('LSTM_CFG', x.shape)
        sz_b, sz_func, sz_bb, sz_token = x.size(0), x.size(1), x.size(2), x.size(3)
        print_memory_occupy('0a')
        x = x.view(-1, x.size(3), x.size(4))
        print_memory_occupy('0b')
        x, (hn,cn) = self.lstm(x)
        print_memory_occupy('0c')

        x = x.view(sz_b, sz_func, sz_bb, sz_token, -1)
        x = x.mean(-2)
        x = self.fc_out(x)
        return x

class LSTM_DFG(nn.Module): #废弃
    def __init__(self, in_dim):
        super(LSTM_DFG, self).__init__()
        self.lstm = nn.LSTM(input_size=in_dim,hidden_size=2*in_dim,batch_first=True,bidirectional=True)
        self.fc_out = nn.Linear(4*in_dim, in_dim)
    def forward(self, x):
        # 1->3 2->4 3->5
        #[batch,config.prog_inst_max,config.inst_token_max,config.dfg_arg.init_dim]
        sz_b, sz_inst, sz_token = x.size(0), x.size(1), x.size(2)

        x = x.view(-1, x.size(2), x.size(3))
        x, (hn,cn) = self.lstm(x)

        x = x.view(sz_b, sz_inst, sz_token, -1)
        x = x.mean(-2)
        x = self.fc_out(x)
        return x

class Attention(nn.Module): #废弃
    def __init__(self, in_dim, out_dim):
        super(Attention, self).__init__()
        self.d_k = out_dim
        self.d_v = out_dim 
        self.w_qs = nn.Linear(in_dim, out_dim, bias=False)
        self.w_ks = nn.Linear(in_dim, out_dim, bias=False)
        self.w_vs = nn.Linear(in_dim, out_dim, bias=False)
        
        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)

    def forward(self, x):
        # 1->3 2->4 3->5
        d_k, d_v = self.d_k, self.d_v
        n_head = 1  
        #sz_b, len_q, len_k, len_v = x.size(0), x.size(1), x.size(1), x.size(1)
        sz_b, sz_func, sz_bb, len_q, len_k, len_v = x.size(0), x.size(1), x.size(2), x.size(3), x.size(3), x.size(3)


        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        #x = x.to_dense().view(-1, x.size(3), x.size(4))
        x = x.view(-1, x.size(3), x.size(4))
        #print_memory_occupy('1a')
        q = self.w_qs(x).view(sz_b, sz_func, sz_bb, len_q, n_head, d_k)
        #print_memory_occupy('1b')
        k = self.w_ks(x).view(sz_b, sz_func, sz_bb, len_k, n_head, d_k)
        #print_memory_occupy('1c')
        v = self.w_vs(x).view(sz_b, sz_func, sz_bb, len_v, n_head, d_v)
        #print_memory_occupy('1d')
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(3, 4), k.transpose(3, 4), v.transpose(3, 4)
        #print_memory_occupy('1e')
        q, attn = self.attention(q, k, v)
        #print_memory_occupy('1f')

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q_all = q.transpose(3, 4).contiguous().view(sz_b, sz_func, sz_bb, len_q, -1)
        #print_memory_occupy('1j')

        q = q_all[:,:,:,0,:]
        #del q_all
        
        return q

class ScaledDotProductAttention(nn.Module): #废弃
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.temperature, k.transpose(4, 5))
        #print("attn:",attn.shape)
        attn = F.softmax(attn, dim=-1)
        #print("attn:",attn.shape)
       
        # attn = F.softmax(torch.matmul(q / self.temperature, k.transpose(4, 5)), dim=-1)
        
        output = torch.matmul(attn, v)
        #print("output:",output.shape)
        

        return output, attn

class Attention2(nn.Module): #废弃
    def __init__(self, in_dim, out_dim):
        super(Attention2, self).__init__()
        self.d_k = out_dim
        self.d_v = out_dim 
        self.w_qs = nn.Linear(in_dim, out_dim, bias=False)
        self.w_ks = nn.Linear(in_dim, out_dim, bias=False)
        self.w_vs = nn.Linear(in_dim, out_dim, bias=False)
        
        self.attention = ScaledDotProductAttention2(temperature=self.d_k ** 0.5)

    def forward(self, x):
        # 1->2 2->3 3->4
        d_k, d_v = self.d_k, self.d_v
        n_head = 1  
        #sz_b, len_q, len_k, len_v = x.size(0), x.size(1), x.size(1), x.size(1)
        sz_b, sz_inst, len_q, len_k, len_v = x.size(0), x.size(1), x.size(2), x.size(2), x.size(2)


        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        #x = x.to_dense().view(-1, x.size(2), x.size(3))
        x = x.view(-1, x.size(2), x.size(3))
        q = self.w_qs(x).view(sz_b, sz_inst, len_q, n_head, d_k)
        k = self.w_ks(x).view(sz_b, sz_inst, len_k, n_head, d_k)
        v = self.w_vs(x).view(sz_b, sz_inst, len_v, n_head, d_v)
        
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)
        
        q, attn = self.attention(q, k, v)
        
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(2, 3).contiguous().view(sz_b, sz_inst, len_q, -1)
        q = q[:,:,0,:]

        return q

class ScaledDotProductAttention2(nn.Module): #废弃
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.temperature, k.transpose(3, 4))

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)

        return output, attn

class GINN(nn.Module):
    def __init__(self, config, graphType, ginn_step):
        super(GINN, self).__init__()
        self.ginn_step = ginn_step
        self.task = config.task
        if graphType == "cfg":
            graph_arg = config.cfg_arg
        elif graphType == "cg":
            graph_arg = config.cg_arg
        elif graphType == "dfg":
            graph_arg = config.dfg_arg
        else:
            print("Error: wrong graph type")
        self.device = config.device1
        

        self.ggnn=[]
        for i in range(self.ginn_step-1):
            if graphType == "cg":
                self.ggnn.append(GGNN(config,'cg',1,i+1))
            elif graphType == "dfg":
                self.ggnn.append(GGNN(config,'dfg',1,i+1))
            else:
                print("Error: wrong graph type")

        
        if self.task == 'funcClassify':
            if graphType == "cg":
                self.ggnn.append(GGNN(config,'cg',2,self.ginn_step))
            elif graphType == "dfg":
                self.ggnn.append(GGNN(config,'dfg',2,self.ginn_step))
        if self.task == 'binaryClassify':
            if graphType == "cg":
                self.ggnn.append(GGNN(config,'cg',2,self.ginn_step))
            elif graphType == "dfg":
                self.ggnn.append(GGNN(config,'dfg',2,self.ginn_step))
        if self.task == 'bug':
            if graphType == "cg":
                print("Error: task bug detection doesn't need call graph.")
            elif graphType == "dfg":
                self.ggnn.append(GGNN(config,'dfg',1,self.ginn_step))
        if self.task == 'deadstore':
            if graphType == "cg":
                self.ggnn.append(GGNN(config,'cg',0,self.ginn_step))
            elif graphType == "dfg":
                self.ggnn.append(GGNN(config,'dfg',1,self.ginn_step))
        for layer in self.ggnn:
            layer.float()
            layer.to(self.device)
        self.ln = nn.LayerNorm([graph_arg.node_max,graph_arg.init_dim])
        
    def forward(self, x, a, m, matrix_list, node_map_list):
        #print("GINN Tensor:",x.size(),m.size(),matrix_list.size(),node_map_list.size(),x.dtype, a.dtype, m.dtype, matrix_list.dtype, node_map_list.dtype)
        weight_list = [] #weight实际上没有用
        for i in range(self.ginn_step): #ginn_level_max
            #print_memory_occupy("ginn step("+str(i)+")")
            #x = x.to(self.device)
            #a = a.to(self.device)
            #x=self.ln(x)
            #a=self.ln(a)
            if i > 0:
                m = matrix_list[:,i-1,:,:] #[batch,ginn_level_max,node_num,node_num]
            if i<self.ginn_step-1:
                node_map = node_map_list[:,i,:,:,:] #[batch,ginn_level_max,node_num,node_num,1]
            else:
                node_map = None
            #node_map = node_map_list
            #print('[Debug] ginn step',i, 'shape(x,a,m):',x.shape,a.shape,m.shape)
            a, weight = self.ggnn[i](x,a,m,node_map) #weight:[batch_size,num_node,num_node,1]
            #print_memory_occupy("ginn step("+str(i)+") get ggnn output")
            weight_list.append(weight)
            # if weight != None:
            #     print("ginn1,(a,weight)",a.shape,weight.shape, cuda.memory_allocated(self.device)/1024**2)
            # else:
            #     print("ginn1,(a)",a.shape, cuda.memory_allocated(self.device)/1024**2)
            #a=out.squeeze()  #[batch,num_node,hidden_size]
            x=a
            #print("GINN Tensor(x):",x.size())
            #print_memory_occupy("ginn step("+str(i)+") finish")
        
        if self.task == 'binaryClassify' or self.task == 'funcClassify':
            return a #out
        # else: #还需修改；暂时不考虑
        #     for i in range(self.ginn_step):
        #         # 此时out为k级收缩后的节点向量, should be [batch,num_node,out_dim]
        #         # x_all = x_all.unsqueeze(1) #在第2维前面增加一个维度
        #         # x_all = x_all.expand(-1,self.n_node,-1,-1) #[batch_size,num_node,num_node,hidden_size]
        #         # weight = F.softmax(torch.norm(x_all, p=None, dim=None, keepdim=True, out=None, dtype=None), dim=-1)
        #         # x_all = x_all * node_map * weight #[batch_size,num_node,num_node,1] (0/1)
        #         # x_out = x_new.sum(2) #sum: num_node -> 1 [batch_size,num_node,1,hidden_size]
        #         node_map = node_map_list[:,i,:,:,:] #[batch,ginn_level_max,node_num,node_num,1]
        #         #node_map.squeeze(1) #[batch,node_num,node_num,1]
        #         out = a.unsqueeze(2)
        #         out = out.expand(-1,-1,self.graph_arg.node_max,-1)
        #         out = out * node_map * weight[i] * self.graph_arg.node_max
        #         out = out.sum(1)
        #     return out


        #如果是输出全图，return out；如果是输出子图或者节点，在这里就拆分，子图需要调用聚集的子模型

class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, config, graphType, outType, level): #配置信息，图的类型（cfg/cg/dfg），输出类型（0/1/2），是哪一级GGNN（如果没有GINN，是纯GGNN，则为1）
        super(GGNN, self).__init__()
        if graphType == "cfg":
            graph_arg = config.cfg_arg
        elif graphType == "cg":
            graph_arg = config.cg_arg
        elif graphType == "dfg":
            graph_arg = config.dfg_arg
        else:
            print("Error: wrong graph type")
        # outType:(0:each node; 1:sub graph; 2:whole graph)
        self.level = level
        self.outType = outType
        self.graphType = graphType
        self.hidden_dim = graph_arg.hidden_dim
        if graphType == "dfg" and level > 1: #是dfg且不是第一层，也就是说节点的初始embedding是通过上一轮GGNN生成的，不是BERT生成的原始embedding
            self.annotation_dim = graph_arg.hidden_dim #graph_arg.init_dim 是上一层的hidden_dim
        else:
            self.annotation_dim = graph_arg.init_dim
        self.n_node = graph_arg.node_max
        self.n_edge = 1
        self.n_output = graph_arg.out_dim
        self.n_steps = graph_arg.steps

        self.fc_in = nn.Linear(self.hidden_dim, self.hidden_dim * self.n_edge)
        self.fc_out = nn.Linear(self.hidden_dim, self.hidden_dim * self.n_edge)

        self.gated_update = GatedPropagation(self.hidden_dim, self.n_node, self.n_edge)

        self.dropout = nn.Dropout(p=0.5)

        if self.outType == 2: #全图生成一个embedding
            self.graph_aggregate = GraphFeature(self.hidden_dim, self.n_node, self.n_edge, self.annotation_dim)
            self.fc_output = nn.Linear(self.hidden_dim, self.n_output)
        elif self.outType == 1: #GINN需要，按照压缩图生成的，虽然从n*n压缩为了m*m（n>m）,但是矩阵的维度仍然是n*n（得到m*m后补0），因为是作为下一轮GGNN的输入，GGNN输入的size需要固定
            self.graph_aggregate = GraphFeature2(self.hidden_dim, self.n_node, self.n_edge, self.annotation_dim)
            self.fc_output = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            self.fc1 = nn.Linear(self.hidden_dim+self.annotation_dim, self.hidden_dim)
            self.fc2 = nn.Linear(self.hidden_dim, 1)
            self.tanh = nn.Tanh()

    def forward(self, x, a, m, node_map):
        '''
        init state x: [batch_size, num_node, hidden_size] , pad zero from annoatation
        annoatation x: [batch_size, num_node, 1] 
        adj matrix m: [batch_size, num_node, num_node * n_edge_types * 2]
        output out: [batch_size, n_label_types], for task 4, 15, 16, n_label_types == num_nodes
        '''
        x, a, m = x.float(), a.float(), m.float()
        #print("config: hidden_dim,annotation_dim,n_output",self.hidden_dim,self.annotation_dim,self.n_output)
        #print("GGNN(x, a, m):",x.shape, a.shape, m.shape)
        #print("model:",self.fc_in,next(self.fc_in.parameters()).is_cuda)
        #all_x = [] # used for task 19, to track 
        #多种类型边：fc_in先将维度为dim的节点向量通过全连接层变为n_edge*dim，再输入gated_update，gated_update模型中会与邻接矩阵计算矩阵乘，得到维度为dim的节点向量
        #所以fc_in中的参数学习了一个节点对应不同类型边的参数，之后通过view，bmm完成形变和计算，变回dim的维度；n_edge合为一的过程为矩阵乘的过程，可以理解为n_edge个向量加起来
        for i in range(self.n_steps):
            in_states = self.fc_in(x)
            out_states = self.fc_out(x)
            #print("GGNN(in_states, out_states, x, m):",in_states.shape, out_states.shape, x.shape, m.shape)
            in_states = in_states.view(-1,self.n_node,self.hidden_dim,self.n_edge).transpose(2,3).transpose(1,2).contiguous()
            in_states = in_states.view(-1, self.n_node*self.n_edge, self.hidden_dim)
            out_states = out_states.view(-1,self.n_node,self.hidden_dim,self.n_edge).transpose(2,3).transpose(1,2).contiguous()
            out_states = out_states.view(-1, self.n_node*self.n_edge, self.hidden_dim)
            #print("GGNN(in_states, out_states, x, m):",in_states.shape, out_states.shape, x.shape, m.shape)
            x = self.gated_update(in_states, out_states, x, m)

        #print("GGNN(x, a):",x.size(), a.size())
        if self.outType == 2:
            #print_memory_occupy("ggnn out2 before graph_aggregate")
            output = self.graph_aggregate(torch.cat((x, a), 2))
            #print_memory_occupy("ggnn out2 after graph_aggregate")
            output = self.fc_output(output)
            weight = None
        elif self.outType == 1:
            #print_memory_occupy("ggnn out1 before graph_aggregate")
            output, weight = self.graph_aggregate(torch.cat((x, a), 2),node_map)
            output = self.fc_output(output)
            #output = self.dropout(output)
            #print_memory_occupy("ggnn out1 after graph_aggregate")
            #output = self.fc_output(output) #size不对，全连接层的输入与输出都是二维张量，一般形状为[batch_size, size]
        else: #self.outType == 0
            output = self.fc1(torch.cat((x, a), 2))
            # output = self.tanh(output)
            # output = self.fc2(output).sum(2)
            output = output[:,0,:] #deadstore任务，CG才属于这类，目标函数为CG的第一个节点，所以只需要返回这个维度的第0个embedding即可
            weight = None
        #print_memory_occupy("ggnn finish")
        return output, weight

class GraphFeature(nn.Module): #get eb of wholw graph
    '''
    Output a Graph-Level Feature
    '''
    def __init__(self, hidden_dim, n_node, n_edge, n_anno):
        super(GraphFeature, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_node = n_node
        self.n_edge = n_edge
        self.n_anno = n_anno

        self.fc_i = nn.Linear(self.hidden_dim + self.n_anno, self.hidden_dim)
        self.fc_j = nn.Linear(self.hidden_dim + self.n_anno, self.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        '''
        input x: [batch_size, num_node, hidden_size + annotation]
        output x: [batch_size, hidden_size]
        '''
        x_sigm = self.sigmoid(self.fc_i(x))
        x_tanh = self.tanh(self.fc_j(x))
        x_new = (x_sigm * x_tanh).sum(1)

        return self.tanh(x_new)

class GraphFeature2(nn.Module): #get eb of all BBs
    '''
    Output a Graph-Level Feature
    '''
    def __init__(self, hidden_dim, n_node, n_edge, n_anno):
        super(GraphFeature2, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_node = n_node
        self.n_edge = n_edge
        self.n_anno = n_anno

        self.fc_i = nn.Linear(self.hidden_dim + self.n_anno, self.hidden_dim)
        self.fc_j = nn.Linear(self.hidden_dim + self.n_anno, self.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def mysoftmax(self,x):
        x_exp = x.exp() #[batch,node_num,node_num]  
        zero = torch.zeros_like(x).float() #[batch,node_num,node_num] 
        x_exp = torch.where(x>0,x_exp,zero)
        partition = x_exp.sum(dim=-1)
        partition = partition.unsqueeze(-1)
        softmax_x = x_exp / partition
        softmax_x2 = torch.where(torch.isnan(softmax_x),zero.float(),softmax_x) #[batch,node_num,node_num] 

        return softmax_x2

    def forward(self, x, node_map):
        '''
        # input x: [batch_size, num_node, hidden_size + annotation]
        # output x: [batch_size, hidden_size]
        input x: [batch_size, num_node, hidden_size + annotation]
        output x: [batch_size, sub_graph_cnt, hidden_size]
        '''
        # 因为要输入下一层，所以节点数依旧是self.n_node，但实际上有效的是sub_graph个节点，剩下的补0，sub_graph由node_map决定
        x_sigm = self.sigmoid(self.fc_i(x))
        x_tanh = self.tanh(self.fc_j(x))
        x_all = x_sigm * x_tanh
        # out of memory
        # x_all = x_all.unsqueeze(1) #在第2维前面增加一个维度
        # x_all = x_all.expand(-1,self.n_node,-1,-1) #[batch_size,num_node,num_node,hidden_size]
        # x_all_norm = torch.norm(x_all, p=2, dim=-1, keepdim=True, out=None, dtype=None)
        # weight = F.softmax(x_all_norm, dim=-1)
        # print("GraphFeature2: x_all,x_all_norm,weight:",x_all.shape,x_all_norm.shape,weight.shape)
        # x_all = x_all * node_map * weight #[batch_size,num_node,num_node,1] (0/1)
        # x_out = x_all.sum(2) #sum: num_node -> 1 [batch_size,num_node,1,hidden_size]
        
        x_all_norm = torch.norm(x_all, p=2, dim=-1, keepdim=True, out=None, dtype=None) #求范数

        x_all_norm = x_all_norm.permute(0,2,1) #[batch,1,node_num]
        node_map = node_map.squeeze() #[batch,node_num,node_num]
        node_map = node_map*x_all_norm #[batch,node_num,node_num]
        weight = self.mysoftmax(node_map) #[batch,node_num,node_num] new weight

        #x_next_level = torch.zeros_like(x_all, requires_grad=True)
        x_all = x_all.unsqueeze(1)
        weight = weight.unsqueeze(-1)
        #print(x_all.shape,weight.shape)
        # for i in range(x_all.size(0)):
        #     x_next_level[i] = (x_all[i] * weight[i]).sum(-2) #[batch,node_num,node_num,hidden_size] -> [batch,node_num,hidden_size]
        # 不用循环会内存不足（调小batch）
        x_all = (x_all * weight).sum(-2) #[batch,node_num,node_num,hidden_size] -> [batch,node_num,hidden_size]
        #print(x_all.shape,weight.shape)

        '''
        # #use a leaf tensor x_next_level
        x_next_level = torch.zeros_like(x_all, requires_grad=True)
        x_all = x_all.unsqueeze(1)
        weight = weight.unsqueeze(-1)
        print(x_all.shape,weight.shape)
        for i in range(x_all.size(0)):
            x_next_level[i] = (x_all[i] * weight[i]).sum(-2) #[batch,node_num,node_num,hidden_size] -> [batch,node_num,hidden_size]
        print(x_next_level.shape,weight.shape)
        '''

        '''
        double loop method to count weight and next_level_node_map
        '''
        # x_next_level = torch.zeros(x_all.size(0),x_all.size(1),x_all.size(2)).float()
        # weight = torch.zeros(x_all.size(0),x_all.size(1),x_all.size(1)).float()

        # for i in range(x_next_level.size(0)):
        #     for j in range(x_next_level.size(1)): #range: num_node
        #         #如果是[batch,num_node,1]
        #         # select_list = torch.zeros(node_map.size(1))
        #         # for k in range(node_map.size(1)):
        #         #     if node_map[i,k]==j:
        #         #         select_list[k]=1

        #         select_list = node_map[i,j,:]
        #         select_list = select_list.view(-1).nonzero().view(-1)
        #         x_all_select = torch.index_select(x_all, 1, select_list)
        #         x_all_norm_t = torch.index_select(x_all_norm, 1, select_list)
        #         x_all_select = x_all_select[i,:,:] #[k,hidden_size]
        #         x_all_norm_t = x_all_norm_t[i,:,:] #[k,1]
        #         x_all_norm_t = F.softmax(x_all_norm_t, dim=-1)
        #         x_all_select = x_all_select*x_all_norm_t
        #         x_all_select = x_all_select.sum(0)
        #         x_next_level[i,j,:] = x_all_select
        #         x_all_norm_t = torch.t(x_all_norm_t)
        #         weight[i,j,0:x_all_norm_t.size(1)] = x_all_norm_t

        #return self.tanh(x_next_level), weight
        return self.tanh(x_all), weight

class GatedPropagation(nn.Module):
    '''
    Gated Recurrent Propagation
    '''
    def __init__(self, hidden_dim, n_node, n_edge):
        super(GatedPropagation, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_node = n_node
        self.n_edge = n_edge

        self.gate_r = nn.Linear(self.hidden_dim*3, self.hidden_dim)
        self.gate_z = nn.Linear(self.hidden_dim*3, self.hidden_dim)
        self.trans  = nn.Linear(self.hidden_dim*3, self.hidden_dim)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x_in, x_out, x_curt, matrix):
        matrix_in  = matrix[:, :, :self.n_node*self.n_edge]
        matrix_out = matrix[:, :, self.n_node*self.n_edge:]
        #print('[Debug] GatedPropagation: x_out,matrix,matrix_out',x_out.shape,matrix.shape,matrix_out.shape)
        a_in  = torch.bmm(matrix_in, x_in) #[batch,n_node,hidden_dim] （维度由n_edge*dim变回dim）
        a_out = torch.bmm(matrix_out, x_out) #[batch,n_node,hidden_dim]
        a = torch.cat((a_in, a_out, x_curt), 2) #[batch,3*n_node,hidden_dim]
        z = self.sigmoid(self.gate_z(a))
        r = self.sigmoid(self.gate_r(a))
        #z = self.relu(self.gate_z(a))
        #r = self.relu(self.gate_r(a))
        #print('[Debug] GatedPropagation: a_in,a_out,r,x_curt',a_in.shape,a_out.shape,r.shape,x_curt.shape)
        joint_input = torch.cat((a_in, a_out, r * x_curt), 2)
        h_hat = self.tanh(self.trans(joint_input))
        output = (1 - z) * x_curt + z * h_hat
        return output
        
class MLP(nn.Module): #废弃
    def __init__(self, input_size, common_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 4, common_size)
        )
                                            
    def forward(self, x):
        out = self.linear(x)
        return out  

class complete(nn.Module):
    def __init__(self,config):
        super(complete, self).__init__()
        self.token_eb_model = config.token_eb_model
        self.task = config.task #任务不同，输出不同
        self.prog_func_max = config.prog_func_max
        if config.task == 'bug':
            self.cfg_ggnn = GGNN(config,'cfg',0,1)
        else:
            self.cfg_ggnn = GGNN(config,'cfg',2,1)
        self.cg_ginn = GINN(config,'cg',config.ginn_max_level)
        self.cg_ggnn = GGNN(config,'cg',2,1)
        self.dfg_ginn = GINN(config,'dfg',config.ginn_max_level)
        self.dfg_ggnn = GGNN(config,'dfg',2,1)
        self.mlp = MLP(config.fc_input_size, config.common_size)
        self.ln = nn.LayerNorm([config.fc_input_size])
        #self.ln_cfg_a = nn.LayerNorm([config.cfg_arg.init_dim])
        self.ln_cfg_a = nn.LayerNorm([config.func_bb_max,config.cfg_arg.init_dim])
        self.ln_cfg_x = nn.LayerNorm([config.cfg_arg.hidden_dim])
        self.ln_dfg_a = nn.LayerNorm([config.dfg_arg.init_dim])
        self.ln_dfg_x = nn.LayerNorm([config.dfg_arg.hidden_dim])
        self.attention_cfg = Attention(config.new_token_eb_dim, config.cfg_arg.init_dim)
        self.attention_dfg = Attention2(config.new_token_eb_dim, config.dfg_arg.init_dim)
        self.lstm_cfg = LSTM_CFG(config.cfg_arg.init_dim)
        self.lstm_dfg = LSTM_DFG(config.dfg_arg.init_dim)
        self.cg_ln = nn.LayerNorm([config.cfg_arg.out_dim])
        if config.task == 'funcClassify':
            if config.subtask == 'compile_class':
                self.fc_out = nn.Linear(config.fc_input_size, config.binary_class)
            else:
                self.fc_out = nn.Linear(config.fc_input_size, config.func_class)
        elif config.task == 'binaryClassify':
            self.fc_out = nn.Linear(config.fc_input_size, config.similarity_class)
        elif config.task == 'bug':
            self.fc_out = nn.Linear(config.fc_input_size, config.bug_class)

    #CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map, label
    #def forward(self,cfg_x_list,cfg_m_list,cfg_a_list,dfg_x,dfg_m,cg_m,dfg_a,node_map,config):
    #def forward(self,CFG_As, CFG_nodes, CFG_nodes_x, cg_A, DFG_nodes, DFG_nodes_x, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map):
    def forward(self,CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map):
        #print("New model forward:",time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time())))
        #print_memory_occupy('before batch train')
        time_list=[]
        time_list.append(time.time())
        # if CFG_As.device != self.device1:
        #     print('Trans to GPU')
        #     CFG_As = CFG_As.to(self.device1) #.to_dense()
        #     CFG_nodes = CFG_nodes.to(self.device1) #.to_dense()
        #     #CFG_nodes_x = CFG_nodes_x.to(self.device1)
        #     cg_A = cg_A.to(self.device1) #.to_dense() 
        #     DFG_nodes = DFG_nodes.to(self.device1) #.to_dense() 
        #     #DFG_nodes_x = DFG_nodes_x.to(self.device3)
        #     dfg_A = dfg_A.to(self.device1) #.to_dense()
        #     DFG_BB_map = DFG_BB_map #.to_sparse() #.to(self.device3)
        #     CG_ginn_matrix = CG_ginn_matrix #.to_sparse() #.to(self.device2)
        #     CG_ginn_node_map = CG_ginn_node_map #.to_sparse() #.to(self.device2)
        #     DFG_ginn_matrix = DFG_ginn_matrix #.to_sparse() #.to(self.device3)
        #     DFG_ginn_node_map = DFG_ginn_node_map #.to_sparse() #.to(self.device3)
        #print("All model Tensor size:",CFG_As.size(), CFG_nodes.size(), cg_A.size(), DFG_nodes.size(), dfg_A.size(), DFG_BB_map.size(), CG_ginn_matrix.size(), CG_ginn_node_map.size(), DFG_ginn_matrix.size(), DFG_ginn_node_map.size(), label.size())
        time_list.append(time.time())

        CFG_nodes = self.ln_cfg_a(CFG_nodes)

        #按GGNN要求padding所有数据，padding规则见padding函数
        #CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A = data_padding(CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A)
        data_padding_result = data_padding(CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_ginn_matrix)
        CFG_As, CFG_nodes, CFG_nodes_x, cg_A, DFG_nodes, DFG_nodes_x, dfg_A, DFG_ginn_matrix = \
            data_padding_result['CFG_As'], \
            data_padding_result['CFG_nodes'], \
            data_padding_result['CFG_nodes_x'], \
            data_padding_result['cg_A'], \
            data_padding_result['DFG_nodes'], \
            data_padding_result['DFG_nodes_x'], \
            data_padding_result['dfg_A'], \
            data_padding_result['dfg_ginn_matrix']

        time_list.append(time.time())
        #print_memory_occupy('0')
        # if self.token_eb_model == 'attention': #实际上这三条if都不会被执行，因为已经通过bert处理好节点初始embedding了
        #     print('[Error] wrong token_eb_model')
        #     CFG_nodes = self.attention_cfg(CFG_nodes) #[config.prog_func_max,config.func_bb_max,config.bb_token_max,config.inst_eb_dim] 先调用attention生成一个cfg.cfg_hidden_dim的向量 
        #     # CFG_nodes: [config.prog_func_max,config.func_bb_max,config.bb_token_max,config.cfg_arg.init_dim] 
        # elif self.token_eb_model == 'lstm':
        #     print('[Error] wrong token_eb_model')
        #     CFG_nodes = self.lstm_cfg(CFG_nodes)
        # elif self.token_eb_model == 'mean':
        #     print('[Error] wrong token_eb_model')
        #     CFG_nodes = CFG_nodes.mean(-2)

        #print_memory_occupy('1')
        #time2 = time.time()
        
        #print_memory_occupy("3")
        this_batch_size=CFG_As.size(0)
        cfg_m = CFG_As.view(-1,CFG_As.size(2),CFG_As.size(3))    #[batch*config.prog_func_max,config.func_bb_max,config.func_bb_max]
        cfg_a = CFG_nodes.view(-1,CFG_nodes.size(2),CFG_nodes.size(3))  #[batch*config.prog_func_max,config.func_bb_max,config.cfg_arg.init_dim]
        #cfg_a = self.ln_cfg_a(cfg_a)
        cfg_x = CFG_nodes_x.view(-1,CFG_nodes_x.size(2),CFG_nodes_x.size(3))
        #cfg_x = self.ln_cfg_x(cfg_x)
        time_list.append(time.time())
        # cfg_m = cfg_m.to(self.device1)
        # cfg_x = cfg_x.to(self.device1)
        # cfg_a = cfg_a.to(self.device1)
        cfg_out, weight = self.cfg_ggnn(cfg_x,cfg_a,cfg_m,None) #should be [batch*config.prog_func_max,config.cfg_arg.out_dim] weight=None
        #cfg_out = self.cg_ln(cfg_out)
        if self.task == 'bug':
            #cfg_out should be [batch*prog_func_max,config.func_bb_max,config.cfg_arg.out_dim]
            #print("cfg_out size:",cfg_out.size())
            cfg_out0 = cfg_out.view(this_batch_size,cfg.new_config.prog_func_max,cfg.new_config.func_bb_max,cfg.new_config.cfg_arg.out_dim)
        else:
            #cfg_out should be [batch*prog_func_max,config.cfg_arg.out_dim]
            #print("cfg_out size:",cfg_out.size())
            cfg_out0 = cfg_out.view(this_batch_size,cfg.new_config.prog_func_max,cfg.new_config.cfg_arg.out_dim)
        time_list.append(time.time())

        # DFG_nodes: [config.prog_inst_max,config.inst_token_max,config.token_eb_dim]
        #print_memory_occupy("9")
        # if self.token_eb_model == 'attention':
        #     print('[Error] wrong token_eb_model')
        #     DFG_nodes = self.attention_dfg(DFG_nodes) #[config.prog_inst_max,config.inst_token_max,config.dfg_arg.init_dim]
        # elif self.token_eb_model == 'lstm':
        #     print('[Error] wrong token_eb_model')
        #     DFG_nodes = self.lstm_dfg(DFG_nodes)
        # elif self.token_eb_model == 'mean':
        #     print('[Error] wrong token_eb_model')
        #     DFG_nodes = DFG_nodes.mean(-2)

        time_list.append(time.time())
        #print_memory_occupy("10")
        #DFG_nodes = DFG_nodes.cpu()torch.cuda.empty_cache()
        #dfg_a = DFG_nodes[:,:,0,:] #[batch,config.prog_inst_max,config.dfg_arg.init_dim]
        dfg_a = DFG_nodes #[batch,config.prog_inst_max,config.dfg_arg.init_dim]
        dfg_x = DFG_nodes_x
        dfg_m = dfg_A
        dfg_ginn_matrix = DFG_ginn_matrix
        dfg_node_map = DFG_ginn_node_map 
        #dfg_a = self.ln_dfg_a(dfg_a)
        #dfg_x = self.ln_dfg_x(dfg_x)
        #print_memory_occupy("before dfg_ginn")
        dfg_out = self.dfg_ginn(dfg_x,dfg_a,dfg_m, dfg_ginn_matrix, dfg_node_map)
        #dfg_out, weight = self.dfg_ggnn(dfg_x,dfg_a,dfg_m,None)
        #print_memory_occupy("after dfg_ginn")
        #print("Tensor:dfg_out",dfg_out.size())
        time_list.append(time.time())
        if self.task != 'bug':
            cg_a = cfg_out0 #should be [batch,config.prog_func_max,config.cg_arg.init_dim] #config.cg_arg.init_dim = config.cfg_arg.out_dim
            #cg_x = cg_a

            this_device = cg_a.device
            padding = torch.zeros(cg_a.shape[0], cg_a.shape[1], cfg.new_config.cg_arg.hidden_dim - cg_a.shape[2]).float().to(this_device)
            cg_x = torch.cat((cg_a, padding), -1)

            cg_m =  cg_A
            cg_ginn_matrix = CG_ginn_matrix
            cg_node_map = CG_ginn_node_map
            #print_memory_occupy("before cg_ginn")
            #cg_out = self.cg_ginn(cg_x,cg_a,cg_m,cg_ginn_matrix,cg_node_map)
            cg_out, weight = self.cg_ggnn(cg_x,cg_a,cg_m,None) #weight=None
            #print_memory_occupy("after cg_ginn")
        time_list.append(time.time())
        time_list_print(time_list)
        
        if self.task == 'funcClassify' or self.task == 'binaryClassify':
            #print("Tensor:dfg_out,cg_out",dfg_out.size(),cg_out.size())
            # cg_out = cg_out.to(self.device_label)
            # dfg_out = dfg_out.to(self.device_label)
            out = torch.cat((dfg_out,cg_out),1)
            #print("Tensor:out",out.size())
            out = self.ln(out)
            #print("Tensor:out",out.size())
            out = self.fc_out(out)
            #print("Tensor:out",out.size())
            #out = F.softmax(out, dim=-1)
            #print("Tensor:out",out.size())
            #print_memory_occupy("21")
            time7 = time.time()
            #print('[Time:]:',round(time2-time1,3),round(time3-time2,3),round(time4-time3,3),round(time5-time4,3),round(time6-time5,3),round(time7-time6,3))

            return out #返回后，在计算loss时，pytorch的实现中还自带一层softmax层，所以我们不自己加softmax

        # elif self.task == 'bug': 
        #     #dfg_out should be [batch,config.prog_inst_max,out_dim]
        #     cfg_out0 = cfg_out0
        #     out = torch.cat((dfg_out,cfg_out0),1)
        #     out = self.ln(out)
        #     out = self.fc_out(out)
        #     #out = F.softmax(out, dim=-1)
        #     return out    

        elif self.task == 'deadstore': #废弃，deadstore实现在别处
            print("[TODO]")    
            #use mlp not fc    

class cfg_bert(nn.Module):
    def __init__(self,config):
        super(cfg_bert, self).__init__()
        self.token_eb_model = config.token_eb_model
        self.task = config.task
        self.prog_func_max = config.prog_func_max
        if config.task == 'bug':
            self.cfg_ggnn = GGNN(config,'cfg',0,1)
        else:
            self.cfg_ggnn = GGNN(config,'cfg',2,1)
        self.ln_cfg_a = nn.LayerNorm([config.func_bb_max,config.cfg_arg.init_dim])
        #self.ln_cfg_a = nn.LayerNorm([config.cfg_arg.init_dim])
        self.ln_cfg_x = nn.LayerNorm([config.cfg_arg.hidden_dim])
        self.mlp = MLP(config.fc_input_size, config.common_size)
        self.ln = nn.LayerNorm([config.fc_input_size])
        self.attention_cfg = Attention(config.new_token_eb_dim, config.cfg_arg.init_dim)
        self.lstm_cfg = LSTM_CFG(config.cfg_arg.init_dim)
        if config.task == 'funcClassify':
            if config.subtask == 'compile_class':
                self.fc_out = nn.Linear(config.fc_input_size, config.binary_class)
            else:
                self.fc_out = nn.Linear(config.fc_input_size, config.func_class)
        elif config.task == 'binaryClassify':
            self.fc_out = nn.Linear(config.fc_input_size, config.similarity_class)
        elif config.task == 'bug':
            self.fc_out = nn.Linear(config.fc_input_size, config.bug_class)

    def forward(self,CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map):
        #print("New model forward:",time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time())))
        #print_memory_occupy('0')
        CFG_As = CFG_As #.to(self.device1) #.to_dense() 
        CFG_nodes = CFG_nodes #.to(self.device1) #.to_dense()
        #CFG_nodes_x = CFG_nodes_x.to(self.device1)
        cg_A = cg_A #.to(self.device2).to_sparse()
        dfg_A = dfg_A #.to(self.device3).to_sparse()
        DFG_BB_map = DFG_BB_map #.to_sparse() #.to(self.device3)
        CG_ginn_matrix = CG_ginn_matrix #.to_sparse() #.to(self.device2)
        CG_ginn_node_map = CG_ginn_node_map #.to_sparse() #.to(self.device2)
        DFG_ginn_matrix = DFG_ginn_matrix #.to_sparse() #.to(self.device3)
        DFG_ginn_node_map = DFG_ginn_node_map #.to_sparse() #.to(self.device3)
        #print("All model Tensor size:",CFG_As.size(), CFG_nodes.size(), cg_A.size(), DFG_nodes.size(), dfg_A.size(), DFG_BB_map.size(), CG_ginn_matrix.size(), CG_ginn_node_map.size(), DFG_ginn_matrix.size(), DFG_ginn_node_map.size(), label.size())

        CFG_nodes = self.ln_cfg_a(CFG_nodes)

        #CFG_As, CFG_nodes, t1, t2, t3 = data_padding(CFG_As, CFG_nodes, None, None, None)
        data_padding_result = data_padding(CFG_As, CFG_nodes, None, None, None, None)
        CFG_As, CFG_nodes, CFG_nodes_x = \
            data_padding_result['CFG_As'], \
            data_padding_result['CFG_nodes'], \
            data_padding_result['CFG_nodes_x']

        # if self.token_eb_model == 'attention':
        #     print('[Error] wrong token_eb_model')
        #     CFG_nodes = self.attention_cfg(CFG_nodes) #[config.prog_func_max,config.func_bb_max,config.bb_token_max,config.inst_eb_dim] 先调用attention生成一个cfg.cfg_hidden_dim的向量 
        #     # CFG_nodes: [config.prog_func_max,config.func_bb_max,config.bb_token_max,config.cfg_arg.init_dim] 
        # elif self.token_eb_model == 'lstm':
        #     print('[Error] wrong token_eb_model')
        #     CFG_nodes = self.lstm_cfg(CFG_nodes)
        # elif self.token_eb_model == 'mean':
        #     print('[Error] wrong token_eb_model')
        #     CFG_nodes = CFG_nodes.mean(-2)
        
        #print_memory_occupy("3")
        this_batch_size=CFG_As.size(0)
        cfg_m = CFG_As.view(-1,CFG_As.size(2),CFG_As.size(3))    #[batch*prog_func_max,config.func_bb_max,config.func_bb_max]
        cfg_a = CFG_nodes.view(-1,CFG_nodes.size(2),CFG_nodes.size(3))  #[batch*prog_func_max,config.func_bb_max,config.cfg_arg.init_dim]
        #cfg_a = self.ln_cfg_a(cfg_a)
        cfg_x = CFG_nodes_x.view(-1,CFG_nodes_x.size(2),CFG_nodes_x.size(3))
        #cfg_x = self.ln_cfg_x(cfg_x)
        # cfg_m = cfg_m.to(self.device1)
        # cfg_x = cfg_x.to(self.device1)
        # cfg_a = cfg_a.to(self.device1)
        cfg_out, weight = self.cfg_ggnn(cfg_x,cfg_a,cfg_m,None) #should be [batch*prog_func_max,config.cfg_arg.out_dim] weight=None
        if self.task == 'bug':
            #cfg_out should be [batch*prog_func_max,config.func_bb_max,config.cfg_arg.out_dim]
            #print("cfg_out size:",cfg_out.size())
            cfg_out0 = cfg_out.view(this_batch_size,cfg.new_config.prog_func_max,cfg.new_config.func_bb_max,cfg.new_config.cfg_arg.out_dim)
        else:
            #cfg_out should be [batch*prog_func_max,config.cfg_arg.out_dim]
            #print("cfg_out size:",cfg_out.size())
            cfg_out0 = cfg_out.view(this_batch_size,cfg.new_config.prog_func_max,cfg.new_config.cfg_arg.out_dim)
         
        if self.task == 'funcClassify' or self.task == 'binaryClassify':
            out = cfg_out0
            #print("Tensor:out",out.size())
            out = self.ln(out)
            out = self.fc_out(out).sum(1)
            #out = F.softmax(out, dim=-1)
            #print("Tensor:out",out.size())
            #print_memory_occupy("21")
            return out

        elif self.task == 'bug':
            #dfg_out should be [batch,config.prog_inst_max,out_dim]
            cfg_out0 = cfg_out0
            out = cfg_out0
            out = self.ln(out)
            out = self.fc_out(out)
            #out = F.softmax(out, dim=-1)
            return out    

        elif self.task == 'deadstore':
            print("[TODO]")    
            #use mlp not fc    

class dfg_bert(nn.Module):
    def __init__(self,config):
        super(dfg_bert, self).__init__()
        self.token_eb_model = config.token_eb_model
        self.task = config.task
        self.prog_func_max = config.prog_func_max
        #self.ln_dfg_a = nn.LayerNorm([config.prog_inst_max, config.dfg_arg.init_dim])
        self.ln_dfg_a = nn.LayerNorm([config.dfg_arg.init_dim])
        #self.ln_dfg_x = nn.LayerNorm([config.dfg_arg.hidden_dim])
        self.dfg_ginn = GINN(config,'dfg',config.ginn_max_level)
        self.dfg_ggnn = GGNN(config,'dfg',2,1)
        self.mlp = MLP(config.fc_input_size, config.common_size)
        self.ln = nn.LayerNorm([config.fc_input_size])
        self.attention_dfg = Attention2(config.new_token_eb_dim, config.dfg_arg.init_dim)
        self.lstm_dfg = LSTM_DFG(config.dfg_arg.init_dim)
        if config.task == 'funcClassify':
            if config.subtask == 'compile_class':
                self.fc_out = nn.Linear(config.fc_input_size, config.binary_class)
            else:
                self.fc_out = nn.Linear(config.fc_input_size, config.func_class)
        elif config.task == 'binaryClassify':
            self.fc_out = nn.Linear(config.fc_input_size, config.similarity_class)
        elif config.task == 'bug':
            self.fc_out = nn.Linear(config.fc_input_size, config.bug_class)

    #CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map, label
    #def forward(self,cfg_x_list,cfg_m_list,cfg_a_list,dfg_x,dfg_m,cg_m,dfg_a,node_map,config):
    #def forward(self,CFG_As, CFG_nodes, CFG_nodes_x, cg_A, DFG_nodes, DFG_nodes_x, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map):
    def forward(self,CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map):
        #print("New model forward:",time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time())))
        print_memory_occupy('before batch train')
        time1 = time.time()
        DFG_nodes = DFG_nodes #.to(self.device3) #.to_dense()
        #DFG_nodes_x = DFG_nodes_x.to(self.device3) #.to_dense()
        dfg_A = dfg_A  #.to(self.device3) #.to_dense()
        DFG_BB_map = DFG_BB_map #.to(self.device3).to_sparse()
        DFG_ginn_matrix = DFG_ginn_matrix #.to(self.device3).to_sparse()
        DFG_ginn_node_map = DFG_ginn_node_map #.to_sparse() #.to(self.device3)


        DFG_nodes = self.ln_dfg_a(DFG_nodes) #归一化

        # DFG_nodes: [config.prog_inst_max,config.inst_token_max,config.token_eb_dim]
        #print_memory_occupy("9")
        # if self.token_eb_model == 'attention':
        #     print('[Error] wrong token_eb_model')
        #     DFG_nodes = self.attention_dfg(DFG_nodes) #[config.prog_inst_max,config.inst_token_max,config.dfg_arg.init_dim]
        # elif self.token_eb_model == 'lstm':
        #     print('[Error] wrong token_eb_model')
        #     DFG_nodes = self.lstm_dfg(DFG_nodes)
        # elif self.token_eb_model == 'mean':
        #     print('[Error] wrong token_eb_model')
        #     DFG_nodes = DFG_nodes.mean(-2)

        #t1, t2, t3, DFG_nodes, dfg_A = data_padding(None, None, None, DFG_nodes, dfg_A)
        data_padding_result = data_padding(None, None, None, DFG_nodes, dfg_A, DFG_ginn_matrix)
        DFG_nodes, DFG_nodes_x, dfg_A, DFG_ginn_matrix = \
            data_padding_result['DFG_nodes'], \
            data_padding_result['DFG_nodes_x'], \
            data_padding_result['dfg_A'], \
            data_padding_result['dfg_ginn_matrix']

        time2 = time.time()

        #print_memory_occupy("10")
        #DFG_nodes = DFG_nodes.cpu()
        #dfg_a = DFG_nodes[:,:,0,:] #[batch,config.prog_inst_max,config.dfg_arg.init_dim]
        
        dfg_a = DFG_nodes #[batch,config.prog_inst_max,config.dfg_arg.init_dim]
        #dfg_a = DFG_nodes_x #for ginn 下一层输入是上一层的hidden_dim，所以不用init的维度，之接都用hidden_dim

        dfg_x = DFG_nodes_x
        dfg_m = dfg_A
        dfg_ginn_matrix = DFG_ginn_matrix
        dfg_node_map = DFG_ginn_node_map 
        #dfg_a = self.ln_dfg_a(dfg_a)
        #dfg_x = self.ln_dfg_x(dfg_x)
        print_memory_occupy("before dfg_ginn")
        
        dfg_out = self.dfg_ginn(dfg_x,dfg_a,dfg_m, dfg_ginn_matrix, dfg_node_map)
        #dfg_out, weight = self.dfg_ggnn(dfg_x,dfg_a,dfg_m,None)
        
        print_memory_occupy("after dfg_ginn")
        #print("Tensor:dfg_out",dfg_out.size())
        time3 = time.time()
        
        if self.task == 'funcClassify' or self.task == 'binaryClassify':
            dfg_out = dfg_out
            out = dfg_out
            #print("Tensor:out",out.size())
            out = self.ln(out)
            #print("Tensor:out",out.size())
            out = self.fc_out(out)
            #print("Tensor:out",out.size())
            #out = F.softmax(out, dim=-1)
            #print("Tensor:out",out.size())
            #print_memory_occupy("21")
            time4 = time.time()
            #print('[Time]:',round(time2-time1,3),round(time3-time2,3),round(time4-time3,3))
            return out

        elif self.task == 'bug':
            #dfg_out should be [batch,config.prog_inst_max,out_dim]
            out = dfg_out
            out = self.ln(out)
            out = self.fc_out(out)
            #out = F.softmax(out, dim=-1)
            return out    

        elif self.task == 'deadstore':
            print("[TODO]")    
            #use mlp not fc    

class cg_bert(nn.Module):
    def __init__(self,config):
        super(cg_bert, self).__init__()
        self.token_eb_model = config.token_eb_model
        self.task = config.task
        self.prog_func_max = config.prog_func_max
        if config.task == 'bug':
            self.cfg_ggnn = GGNN(config,'cfg',0,1)
        else:
            self.cfg_ggnn = GGNN(config,'cfg',2,1)
        self.cg_ln = nn.LayerNorm([config.prog_func_max,config.cfg_arg.out_dim])
        self.cg_ginn = GINN(config,'cg',config.ginn_max_level)
        self.cg_ggnn = GGNN(config,'cg',2,1)
        self.mlp = MLP(config.fc_input_size, config.common_size)
        self.ln = nn.LayerNorm([config.fc_input_size])
        self.ln_cfg_a = nn.LayerNorm([config.func_bb_max,config.cfg_arg.init_dim])
        self.ln_cfg_x = nn.LayerNorm([config.cfg_arg.hidden_dim])
        #self.ln_cg_a = nn.LayerNorm([config.prog_func_max,config.cg_arg.hidden_dim])
        self.attention_cfg = Attention(config.new_token_eb_dim, config.cfg_arg.init_dim)
        self.lstm_cfg = LSTM_CFG(config.cfg_arg.init_dim)
        if config.task == 'funcClassify':
            if config.subtask == 'compile_class':
                self.fc_out = nn.Linear(config.fc_input_size, config.binary_class)
            else:
                self.fc_out = nn.Linear(config.fc_input_size, config.func_class)
        elif config.task == 'binaryClassify':
            self.fc_out = nn.Linear(config.fc_input_size, config.similarity_class)
        elif config.task == 'bug':
            self.fc_out = nn.Linear(config.fc_input_size, config.bug_class)

    #CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map, label
    #def forward(self,cfg_x_list,cfg_m_list,cfg_a_list,dfg_x,dfg_m,cg_m,dfg_a,node_map,config):
    #def forward(self,CFG_As, CFG_nodes, CFG_nodes_x, cg_A, DFG_nodes, DFG_nodes_x, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map):
    def forward(self,CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map):
        if torch.any(torch.isnan(CFG_As)) or \
            torch.any(torch.isnan(CFG_nodes)) or \
            torch.any(torch.isnan(cg_A)):
            print('[Error] has NaN') 
        #print("New model forward:",time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time())))
        #print_memory_occupy('before batch train')
        CFG_As = CFG_As #.to(self.device1) #.to_dense()
        CFG_nodes = CFG_nodes #.to(self.device1) #.to_dense()
        #CFG_nodes_x = CFG_nodes_x.to(self.device1)
        cg_A = cg_A #.to(self.device2) #.to_dense()
        CG_ginn_matrix = CG_ginn_matrix #.to_sparse() #.to(self.device2)
        CG_ginn_node_map = CG_ginn_node_map #.to_sparse() #.to(self.device2)

        CFG_nodes = self.ln_cfg_a(CFG_nodes)

        #CFG_As, CFG_nodes, cg_A, t1, t2 = data_padding(CFG_As, CFG_nodes, cg_A, None, None)
        data_padding_result = data_padding(CFG_As, CFG_nodes, cg_A, None, None, None)
        CFG_As, CFG_nodes, CFG_nodes_x, cg_A = \
            data_padding_result['CFG_As'], \
            data_padding_result['CFG_nodes'], \
            data_padding_result['CFG_nodes_x'], \
            data_padding_result['cg_A']

        # #print_memory_occupy('0')
        # if self.token_eb_model == 'attention':
        #     print('[Error] wrong token_eb_model')
        #     CFG_nodes = self.attention_cfg(CFG_nodes) #[config.prog_func_max,config.func_bb_max,config.bb_token_max,config.inst_eb_dim] 先调用attention生成一个cfg.cfg_hidden_dim的向量 
        #     # CFG_nodes: [config.prog_func_max,config.func_bb_max,config.bb_token_max,config.cfg_arg.init_dim] 
        # elif self.token_eb_model == 'lstm':
        #     print('[Error] wrong token_eb_model')
        #     CFG_nodes = self.lstm_cfg(CFG_nodes)
        # elif self.token_eb_model == 'mean':
        #     print('[Error] wrong token_eb_model')
        #     CFG_nodes = CFG_nodes.mean(-2)

        this_batch_size=CFG_As.size(0)
        cfg_m = CFG_As.view(-1,CFG_As.size(2),CFG_As.size(3))    #[batch*config.prog_func_max,config.func_bb_max,config.func_bb_max]
        cfg_a = CFG_nodes.view(-1,CFG_nodes.size(2),CFG_nodes.size(3))  #[batch*config.prog_func_max,config.func_bb_max,config.cfg_arg.init_dim]
        #cfg_a = self.ln_cfg_a(cfg_a)
        cfg_x = CFG_nodes_x.view(-1,CFG_nodes_x.size(2),CFG_nodes_x.size(3))
        #cfg_x = self.ln_cfg_x(cfg_x)
        # cfg_m = cfg_m.to(self.device1)
        # cfg_x = cfg_x.to(self.device1)
        # cfg_a = cfg_a.to(self.device1)
        cfg_out, weight = self.cfg_ggnn(cfg_x,cfg_a,cfg_m,None) #should be [batch*config.prog_func_max,config.cfg_arg.out_dim] weight=None
        if self.task == 'bug':
            cfg_out0 = cfg_out.view(this_batch_size,cfg.new_config.prog_func_max,cfg.new_config.func_bb_max,cfg.new_config.cfg_arg.out_dim)
        else:
            cfg_out0 = cfg_out.view(this_batch_size,cfg.new_config.prog_func_max,cfg.new_config.cfg_arg.out_dim)
    
        
        if self.task != 'bug':
            cg_a = cfg_out0 #should be [batch,config.prog_func_max,config.cg_arg.init_dim]
            #cg_x = cg_a

            #cg_a = self.cg_ln(cg_a)

            this_device = cg_a.device
            padding = torch.zeros(cg_a.shape[0], cg_a.shape[1], cfg.new_config.cg_arg.hidden_dim - cg_a.shape[2]).float().to(this_device)
            cg_x = torch.cat((cg_a, padding), -1)

            
            cg_m = cg_A
            cg_ginn_matrix = CG_ginn_matrix
            cg_node_map = CG_ginn_node_map
            #print('[Debug] cg_a\n', cg_a)
            #cg_out = self.cg_ginn(cg_x,cg_a,cg_m,cg_ginn_matrix,cg_node_map)
            cg_out, weight = self.cg_ggnn(cg_x,cg_a,cg_m,None)
            #cg_out = cg_a
            #print('[Debug] cg_out\n', cg_out)
            debug_out = self.fc_out(cg_out).sum(1)
            #print('[Debug] debug_out\n', debug_out)
        
        if self.task == 'funcClassify' or self.task == 'binaryClassify':
            out = cg_out
            out = self.ln(out)
            out = self.fc_out(out)   
            #print_memory_occupy("21")
            return out

        elif self.task == 'bug':
            cfg_out0 = cfg_out0
            out = cfg_out0
            out = self.ln(out)
            out = self.fc_out(out)
            #out = F.softmax(out, dim=-1)
            return out    

        elif self.task == 'deadstore':
            print("[TODO]")    
            #use mlp not fc    

class cg_only(nn.Module): #废弃
    def __init__(self,config):
        super(cg_only, self).__init__()
        self.token_eb_model = config.token_eb_model
        self.task = config.task
        self.prog_func_max = config.prog_func_max
        self.cg_ginn = GINN(config,'cg',config.ginn_max_level)
        self.cg_ggnn = GGNN(config,'cg',2,1)
        self.mlp = MLP(config.fc_input_size, config.common_size)
        self.ln = nn.LayerNorm([config.fc_input_size])
        #self.ln_cg_a = nn.LayerNorm([config.prog_func_max,config.cg_arg.hidden_dim])
        if config.task == 'funcClassify':
            if config.subtask == 'compile_class':
                self.fc_out = nn.Linear(config.fc_input_size, config.binary_class)
            else:
                self.fc_out = nn.Linear(config.fc_input_size, config.func_class)
        elif config.task == 'binaryClassify':
            self.fc_out = nn.Linear(config.fc_input_size, config.similarity_class)
        elif config.task == 'bug':
            self.fc_out = nn.Linear(config.fc_input_size, config.bug_class)

    #CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map, label
    #def forward(self,cfg_x_list,cfg_m_list,cfg_a_list,dfg_x,dfg_m,cg_m,dfg_a,node_map,config):
    #def forward(self,CFG_As, CFG_nodes, CFG_nodes_x, cg_A, DFG_nodes, DFG_nodes_x, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map):
    def forward(self,CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map):
    
        #print("New model forward:",time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time())))
        #print_memory_occupy('before batch train')
        CFG_As = CFG_As #.to(self.device1) #.to_dense()
        CFG_nodes = CFG_nodes #.to(self.device1) #.to_dense()
        #CFG_nodes_x = CFG_nodes_x.to(self.device1)
        cg_A = cg_A #.to(self.device2) #.to_dense()
        CG_ginn_matrix = CG_ginn_matrix #.to_sparse() #.to(self.device2)
        CG_ginn_node_map = CG_ginn_node_map #.to_sparse() #.to(self.device2)

        #CFG_As, CFG_nodes, cg_A, t1, t2 = data_padding(CFG_As, CFG_nodes, cg_A, None, None)
        data_padding_result = data_padding(CFG_As, CFG_nodes, cg_A, None, None, None)
        CFG_As, CFG_nodes, CFG_nodes_x, cg_A = \
            data_padding_result['CFG_As'], \
            data_padding_result['CFG_nodes'], \
            data_padding_result['CFG_nodes_x'], \
            data_padding_result['cg_A']
            


        cfg_out0 = torch.zeros((cg_A.size(0),cfg.new_config.prog_func_max,cfg.new_config.cg_arg.init_dim)).float().to(cfg.new_config.device1)
        
        if self.task != 'bug':
            cfg_out0 = cfg_out0
            cg_a = cfg_out0 #should be [batch,config.prog_func_max,config.cg_arg.init_dim]
            cg_x = cg_a
            cg_m = cg_A
            cg_ginn_matrix = CG_ginn_matrix
            cg_node_map = CG_ginn_node_map
            #print_memory_occupy("before cg_ginn")
            #cg_out = self.cg_ginn(cg_x,cg_a,cg_m,cg_ginn_matrix,cg_node_map)
            cg_out, weight = self.cg_ggnn(cg_x,cg_a,cg_m,None)
            #print_memory_occupy("after cg_ginn")
        
        if self.task == 'funcClassify' or self.task == 'binaryClassify':
            cg_out = cg_out
            out = cg_out
            out = self.ln(out)
            out = self.fc_out(out)
            #print_memory_occupy("21")
            return out

        elif self.task == 'bug':
            cfg_out0 = cfg_out0
            out = cfg_out0
            out = self.ln(out)
            out = self.fc_out(out)
            #out = F.softmax(out, dim=-1)
            return out    

        elif self.task == 'deadstore':
            print("[TODO]")    
            #use mlp not fc    

class no_cg(nn.Module): #废弃
    def __init__(self,config):
        super(no_cg, self).__init__()
        self.token_eb_model = config.token_eb_model
        self.task = config.task
        self.prog_func_max = config.prog_func_max
        if config.task == 'bug':
            self.cfg_ggnn = GGNN(config,'cfg',0,1)
        else:
            self.cfg_ggnn = GGNN(config,'cfg',2,1)
        self.dfg_ginn = GINN(config,'dfg',config.ginn_max_level)
        self.dfg_ggnn = GGNN(config,'dfg',2,1)
        self.ln = nn.LayerNorm([config.fc_input_size])
        #self.ln_cfg_a = nn.LayerNorm([config.cfg_arg.init_dim])
        self.ln_cfg_a = nn.LayerNorm([config.func_bb_max,config.cfg_arg.init_dim])
        self.ln_cfg_x = nn.LayerNorm([config.cfg_arg.hidden_dim])
        self.ln_dfg_a = nn.LayerNorm([config.dfg_arg.init_dim])
        self.ln_dfg_x = nn.LayerNorm([config.dfg_arg.hidden_dim])
        self.attention_cfg = Attention(config.new_token_eb_dim, config.cfg_arg.init_dim)
        self.attention_dfg = Attention2(config.new_token_eb_dim, config.dfg_arg.init_dim)
        self.lstm_cfg = LSTM_CFG(config.cfg_arg.init_dim)
        self.lstm_dfg = LSTM_DFG(config.dfg_arg.init_dim)
        self.cg_ln = nn.LayerNorm([config.cfg_arg.out_dim])
        if config.task == 'funcClassify':
            if config.subtask == 'compile_class':
                self.fc_out = nn.Linear(config.fc_input_size, config.binary_class)
            else:
                self.fc_out = nn.Linear(config.fc_input_size, config.func_class)
        elif config.task == 'binaryClassify':
            self.fc_out = nn.Linear(config.fc_input_size, config.similarity_class)
        elif config.task == 'bug':
            self.fc_out = nn.Linear(config.fc_input_size, config.bug_class)

    #CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map, label
    #def forward(self,cfg_x_list,cfg_m_list,cfg_a_list,dfg_x,dfg_m,cg_m,dfg_a,node_map,config):
    #def forward(self,CFG_As, CFG_nodes, CFG_nodes_x, cg_A, DFG_nodes, DFG_nodes_x, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map):
    def forward(self,CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map):
        #print("New model forward:",time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time())))
        #print_memory_occupy('before batch train')
        time_list=[]
        time_list.append(time.time())

        CFG_nodes = self.ln_cfg_a(CFG_nodes)

        #CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A = data_padding(CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A)
        data_padding_result = data_padding(CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_ginn_matrix)
        CFG_As, CFG_nodes, CFG_nodes_x, cg_A, DFG_nodes, DFG_nodes_x, dfg_A, DFG_ginn_matrix = \
            data_padding_result['CFG_As'], \
            data_padding_result['CFG_nodes'], \
            data_padding_result['CFG_nodes_x'], \
            data_padding_result['cg_A'], \
            data_padding_result['DFG_nodes'], \
            data_padding_result['DFG_nodes_x'], \
            data_padding_result['dfg_A'], \
            data_padding_result['dfg_ginn_matrix']

        time_list.append(time.time())
        #print_memory_occupy('0')
        if self.token_eb_model == 'attention':
            print('[Error] wrong token_eb_model')
            CFG_nodes = self.attention_cfg(CFG_nodes) #[config.prog_func_max,config.func_bb_max,config.bb_token_max,config.inst_eb_dim] 先调用attention生成一个cfg.cfg_hidden_dim的向量 
            # CFG_nodes: [config.prog_func_max,config.func_bb_max,config.bb_token_max,config.cfg_arg.init_dim] 
        elif self.token_eb_model == 'lstm':
            print('[Error] wrong token_eb_model')
            CFG_nodes = self.lstm_cfg(CFG_nodes)
        elif self.token_eb_model == 'mean':
            print('[Error] wrong token_eb_model')
            CFG_nodes = CFG_nodes.mean(-2)

        #print_memory_occupy('1')
        #time2 = time.time()
        
        #print_memory_occupy("3")
        this_batch_size=CFG_As.size(0)
        cfg_m = CFG_As.view(-1,CFG_As.size(2),CFG_As.size(3))    #[batch,config.func_bb_max,config.func_bb_max]
        cfg_a = CFG_nodes.view(-1,CFG_nodes.size(2),CFG_nodes.size(3))  #[batch,config.func_bb_max,config.cfg_arg.init_dim]
        #cfg_a = self.ln_cfg_a(cfg_a)
        cfg_x = CFG_nodes_x.view(-1,CFG_nodes_x.size(2),CFG_nodes_x.size(3))
        #cfg_x = self.ln_cfg_x(cfg_x)
        time_list.append(time.time())
        # cfg_m = cfg_m.to(self.device1)
        # cfg_x = cfg_x.to(self.device1)
        # cfg_a = cfg_a.to(self.device1)
        cfg_out, weight = self.cfg_ggnn(cfg_x,cfg_a,cfg_m,None) #should be [batch,config.cfg_arg.out_dim] weight=None
        #cfg_out = self.cg_ln(cfg_out)
        if self.task == 'bug':
            #cfg_out should be [batch*prog_func_max,config.func_bb_max,config.cfg_arg.out_dim]
            #print("cfg_out size:",cfg_out.size())
            cfg_out0 = cfg_out.view(this_batch_size,cfg.new_config.prog_func_max,cfg.new_config.func_bb_max,cfg.new_config.cfg_arg.out_dim)
        else:
            #cfg_out should be [batch*prog_func_max,config.cfg_arg.out_dim]
            #print("cfg_out size:",cfg_out.size())
            cfg_out0 = cfg_out.view(this_batch_size,cfg.new_config.prog_func_max,cfg.new_config.cfg_arg.out_dim)
        time_list.append(time.time())

        # DFG_nodes: [config.prog_inst_max,config.inst_token_max,config.token_eb_dim]
        #print_memory_occupy("9")
        if self.token_eb_model == 'attention':
            print('[Error] wrong token_eb_model')
            DFG_nodes = self.attention_dfg(DFG_nodes) #[config.prog_inst_max,config.inst_token_max,config.dfg_arg.init_dim]
        elif self.token_eb_model == 'lstm':
            print('[Error] wrong token_eb_model')
            DFG_nodes = self.lstm_dfg(DFG_nodes)
        elif self.token_eb_model == 'mean':
            print('[Error] wrong token_eb_model')
            DFG_nodes = DFG_nodes.mean(-2)

        time_list.append(time.time())
        #print_memory_occupy("10")
        #DFG_nodes = DFG_nodes.cpu()torch.cuda.empty_cache()
        #dfg_a = DFG_nodes[:,:,0,:] #[batch,config.prog_inst_max,config.dfg_arg.init_dim]
        dfg_a = DFG_nodes #[batch,config.prog_inst_max,config.dfg_arg.init_dim]
        dfg_x = DFG_nodes_x
        dfg_m = dfg_A
        dfg_ginn_matrix = DFG_ginn_matrix
        dfg_node_map = DFG_ginn_node_map 
        dfg_out = self.dfg_ginn(dfg_x,dfg_a,dfg_m, dfg_ginn_matrix, dfg_node_map)
        
        time_list.append(time.time())
        time_list_print(time_list)
        
        if self.task == 'funcClassify' or self.task == 'binaryClassify':
            cfg_out = cfg_out0.sum(1)
            out = torch.cat((dfg_out,cfg_out),1)
            #print("Tensor:out",out.size())
            out = self.ln(out)
            #print("Tensor:out",out.size())
            out = self.fc_out(out)
            time7 = time.time()
            #print('[Time:]:',round(time2-time1,3),round(time3-time2,3),round(time4-time3,3),round(time5-time4,3),round(time6-time5,3),round(time7-time6,3))

            return out

        elif self.task == 'bug':
            #dfg_out should be [batch,config.prog_inst_max,out_dim]
            cfg_out0 = cfg_out0
            out = torch.cat((dfg_out,cfg_out0),1)
            out = self.ln(out)
            out = self.fc_out(out)
            #out = F.softmax(out, dim=-1)
            return out    

        elif self.task == 'deadstore':
            print("[TODO]")    
            #use mlp not fc    

class dead_cg(nn.Module):
    def __init__(self,config):
        super(dead_cg, self).__init__()
        self.token_eb_model = config.token_eb_model
        self.task = config.task
        self.prog_func_max = config.prog_func_max
        self.cfg_ggnn = GGNN(config,'cfg',2,1) #config, graphType, outType, level
        self.cg_ln = nn.LayerNorm([config.cfg_arg.out_dim])
        self.cg_ginn = GINN(config,'cg',config.ginn_max_level)
        self.cg_ggnn = GGNN(config,'cg',0,1)
        self.mlp = MLP(config.fc_input_size, config.common_size)
        self.ln = nn.LayerNorm([config.fc_input_size])
        self.ln_cfg_a = nn.LayerNorm([config.func_bb_max,config.cfg_arg.init_dim])
        self.ln_cfg_x = nn.LayerNorm([config.cfg_arg.hidden_dim])
        #self.ln_cg_a = nn.LayerNorm([config.prog_func_max,config.cg_arg.hidden_dim])
        self.attention_cfg = Attention(config.new_token_eb_dim, config.cfg_arg.init_dim)
        self.lstm_cfg = LSTM_CFG(config.cfg_arg.init_dim)
        self.fc_out = nn.Linear(config.fc_input_size, 2)

    #CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map, label
    #def forward(self,cfg_x_list,cfg_m_list,cfg_a_list,dfg_x,dfg_m,cg_m,dfg_a,node_map,config):
    #def forward(self,CFG_As, CFG_nodes, CFG_nodes_x, cg_A, DFG_nodes, DFG_nodes_x, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map):
    def forward(self,CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map):
        if torch.any(torch.isnan(CFG_As)) or \
            torch.any(torch.isnan(CFG_nodes)) or \
            torch.any(torch.isnan(cg_A)):
            print('[Error] has NaN') 

        CFG_nodes = self.ln_cfg_a(CFG_nodes) #归一化

        #按照GGNN的输入进行padding(详细规则见最下方padding函数，实际实现的功能就是下方padding函数的功能)
        cfg_padding = CFG_As.permute(0,1,3,2)
        CFG_As = torch.cat((CFG_As,cfg_padding),3)

        this_device = DFG_nodes.device
        padding = torch.zeros(CFG_nodes.shape[0], CFG_nodes.shape[1], CFG_nodes.shape[2], cfg.new_config.cfg_arg.hidden_dim - CFG_nodes.shape[3]).float().to(this_device)
        CFG_nodes_x = torch.cat((CFG_nodes, padding), -1)

        cg_padding = cg_A.permute(0,2,1)
        cg_A = torch.cat((cg_A,cg_padding),2)

        # data_padding_result = data_padding(CFG_As, CFG_nodes, cg_A, None, None, None)
        # CFG_As, CFG_nodes, CFG_nodes_x, cg_A = \
        #     data_padding_result['CFG_As'], \
        #     data_padding_result['CFG_nodes'], \
        #     data_padding_result['CFG_nodes_x'], \
        #     data_padding_result['cg_A']

        # #print_memory_occupy('0')
        # if self.token_eb_model == 'attention':
        #     print('[Error] wrong token_eb_model')
        #     CFG_nodes = self.attention_cfg(CFG_nodes) #[config.prog_func_max,config.func_bb_max,config.bb_token_max,config.inst_eb_dim] 先调用attention生成一个cfg.cfg_hidden_dim的向量 
        #     # CFG_nodes: [config.prog_func_max,config.func_bb_max,config.bb_token_max,config.cfg_arg.init_dim] 
        # elif self.token_eb_model == 'lstm':
        #     print('[Error] wrong token_eb_model')
        #     CFG_nodes = self.lstm_cfg(CFG_nodes)
        # elif self.token_eb_model == 'mean':
        #     print('[Error] wrong token_eb_model')
        #     CFG_nodes = CFG_nodes.mean(-2)

        this_batch_size=CFG_As.size(0)
        cfg_m = CFG_As.view(-1,CFG_As.size(2),CFG_As.size(3))    #[batch*program_func_max,config.func_bb_max,config.func_bb_max]
        cfg_a = CFG_nodes.view(-1,CFG_nodes.size(2),CFG_nodes.size(3))  #[batch*program_func_max,config.func_bb_max,config.cfg_arg.init_dim]
        #cfg_a = self.ln_cfg_a(cfg_a)
        cfg_x = CFG_nodes_x.view(-1,CFG_nodes_x.size(2),CFG_nodes_x.size(3))
        #cfg_x = self.ln_cfg_x(cfg_x)
        # cfg_m = cfg_m.to(self.device1)
        # cfg_x = cfg_x.to(self.device1)
        # cfg_a = cfg_a.to(self.device1)
        cfg_out, weight = self.cfg_ggnn(cfg_x,cfg_a,cfg_m,None) #should be [batch*program_func_max,config.cfg_arg.out_dim] weight=None
        #print('[DEBUG] cfg_out size:',cfg_out.shape)
        cfg_out0 = cfg_out.view(this_batch_size,cfg.new_config.prog_func_max,cfg.new_config.cfg_arg.out_dim)
        

        cg_a = cfg_out0 #should be [batch,config.prog_func_max,config.cfg_arg.out_dim]
        cg_a = self.cg_ln(cg_a)
        this_device = cg_a.device
        padding = torch.zeros(cg_a.shape[0], cg_a.shape[1], cfg.new_config.cg_arg.hidden_dim - cg_a.shape[2]).float().to(this_device)
        cg_x = torch.cat((cg_a, padding), -1)
        
        cg_m = cg_A
        cg_out, weight = self.cg_ggnn(cg_x,cg_a,cg_m,None) #返回的已经是第一个节点(目标函数）的embedding
        
        out = cg_out
        out = self.ln(out)
        #print('out size:',out.shape)
        out = self.fc_out(out)
        #print_memory_occupy("21")
        return out
 
class dead_cfg(nn.Module): #废弃
    def __init__(self,config):
        super(dead_cfg, self).__init__()
        self.token_eb_model = config.token_eb_model
        self.task = config.task
        self.prog_func_max = config.prog_func_max
        self.cfg_ggnn = GGNN(config,'cfg',2,1) #config, graphType, outType, level
        self.cg_ln = nn.LayerNorm([config.prog_func_max,config.cfg_arg.out_dim])
        self.cg_ginn = GINN(config,'cg',config.ginn_max_level)
        self.cg_ggnn = GGNN(config,'cg',0,1)
        self.mlp = MLP(config.fc_input_size, config.common_size)
        self.ln = nn.LayerNorm([config.fc_input_size])
        self.ln_cfg_a = nn.LayerNorm([config.func_bb_max,config.cfg_arg.init_dim])
        self.ln_cfg_x = nn.LayerNorm([config.cfg_arg.hidden_dim])
        #self.ln_cg_a = nn.LayerNorm([config.prog_func_max,config.cg_arg.hidden_dim])
        self.attention_cfg = Attention(config.new_token_eb_dim, config.cfg_arg.init_dim)
        self.lstm_cfg = LSTM_CFG(config.cfg_arg.init_dim)
        self.fc_out = nn.Linear(config.fc_input_size, 2)

    #CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map, label
    #def forward(self,cfg_x_list,cfg_m_list,cfg_a_list,dfg_x,dfg_m,cg_m,dfg_a,node_map,config):
    #def forward(self,CFG_As, CFG_nodes, CFG_nodes_x, cg_A, DFG_nodes, DFG_nodes_x, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map):
    def forward(self,CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A, DFG_BB_map, CG_ginn_matrix, CG_ginn_node_map, DFG_ginn_matrix, DFG_ginn_node_map):
        if torch.any(torch.isnan(CFG_As)) or \
            torch.any(torch.isnan(CFG_nodes)) or \
            torch.any(torch.isnan(cg_A)):
            print('[Error] has NaN') 

        CFG_As = CFG_As[:,0,:,:]
        CFG_nodes = CFG_nodes[:,0,:,:]

        CFG_nodes = self.ln_cfg_a(CFG_nodes)

        cfg_padding = CFG_As.permute(0,2,1)
        CFG_As = torch.cat((CFG_As,cfg_padding),2)

        this_device = DFG_nodes.device
        padding = torch.zeros(CFG_nodes.shape[0], CFG_nodes.shape[1], cfg.new_config.cfg_arg.hidden_dim - CFG_nodes.shape[2]).float().to(this_device)
        CFG_nodes_x = torch.cat((CFG_nodes, padding), -1)
        cfg_x = CFG_nodes_x
        cfg_a = CFG_nodes
        cfg_m = CFG_As


        cfg_out, weight = self.cfg_ggnn(cfg_x,cfg_a,cfg_m,None) #should be [batch,config.cfg_arg.
        
        out = cfg_out
        print("Tensor:out",out.size())
        out = self.ln(out)
        out = self.fc_out(out)
        return out
 

class model_pair(nn.Module):
    def __init__(self,config):
        super(model_pair, self).__init__() #根据设定初始化模型
        if cfg.new_config.model_type=="cfg_bert":
            self.mf_model = cfg_bert(config)
        elif cfg.new_config.model_type=="cg_bert":
            self.mf_model = cg_bert(config)
        elif cfg.new_config.model_type=="dfg_bert":
            self.mf_model = dfg_bert(config)
        elif cfg.new_config.model_type=="no_cg":
            self.mf_model = no_cg(config)
        elif cfg.new_config.model_type=="all":
            self.mf_model = complete(config)

    def forward(self,CFG_As_a, CFG_nodes_a, cg_A_a, DFG_nodes_a, dfg_A_a, DFG_BB_map_a, CG_ginn_matrix_a, CG_ginn_node_map_a, DFG_ginn_matrix_a, DFG_ginn_node_map_a, \
                CFG_As_b, CFG_nodes_b, cg_A_b, DFG_nodes_b, dfg_A_b, DFG_BB_map_b, CG_ginn_matrix_b, CG_ginn_node_map_b, DFG_ginn_matrix_b, DFG_ginn_node_map_b):
        #print("New model forward:",time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time())))
        out1 = self.mf_model(CFG_As_a, CFG_nodes_a, cg_A_a, DFG_nodes_a, dfg_A_a, DFG_BB_map_a, CG_ginn_matrix_a, CG_ginn_node_map_a, DFG_ginn_matrix_a, DFG_ginn_node_map_a) #第一个样例的embedding
        #print("Model1 forward finish:",time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time())))
        out2 = self.mf_model(CFG_As_b, CFG_nodes_b, cg_A_b, DFG_nodes_b, dfg_A_b, DFG_BB_map_b, CG_ginn_matrix_b, CG_ginn_node_map_b, DFG_ginn_matrix_b, DFG_ginn_node_map_b) #第二个样例的embedding
        #print("Model2 forward finish:",time.strftime('%m_%d_%H_%M_%S',time.localtime(time.time())))
        return out1, out2

def data_padding(CFG_As=None, CFG_nodes=None, cg_A=None, DFG_nodes=None, dfg_A=None, dfg_ginn_matrix=None):
        # #padding
        # if CFG_As != None:
        #     cfg_0_padding = (0,cfg.new_config.func_bb_max - CFG_As.size(-1),\
        #                     0,cfg.new_config.func_bb_max - CFG_As.size(-2),\
        #                     0,cfg.new_config.prog_func_max - CFG_As.size(-3))
        #     CFG_As = F.pad(CFG_As,cfg_0_padding)
        #     cfg_padding = CFG_As.permute(0,2,1)
        #     CFG_As = torch.cat((CFG_As,cfg_padding),2)

        # if CFG_nodes != None:
        #     cfg_nodes_0_padding = (0,0,\
        #                         0,cfg.new_config.bb_token_max - CFG_nodes.size(-2),\
        #                         0,cfg.new_config.func_bb_max - CFG_nodes.size(-3),\
        #                         0,cfg.new_config.prog_func_max - CFG_nodes.size(-4))
        #     CFG_nodes = F.pad(CFG_nodes,cfg_nodes_0_padding)

        # if cg_A != None:
        #     cg_a_0_padding = (0,cfg.new_config.prog_func_max - cg_A.size(-1),\
        #                     0,cfg.new_config.prog_func_max - cg_A.size(-2))
        #     cg_A = F.pad(cg_A,cg_a_0_padding)
        #     cg_padding = cg_A.permute(1,0)
        #     cg_A = torch.cat((cg_A,cg_padding),1)

        # if DFG_nodes != None:
        #     dfg_nodes_0_padding = (0,0,\
        #                         0,cfg.new_config.inst_token_max - DFG_nodes.size(-2),\
        #                         0,cfg.new_config.prog_inst_max - DFG_nodes.size(-3))
        #     DFG_nodes = F.pad(DFG_nodes,dfg_nodes_0_padding)

        # if dfg_A != None:
        #     dfg_a_0_padding = (0,cfg.new_config.prog_inst_max - dfg_A.size(-1),\
        #                     0,cfg.new_config.prog_inst_max - dfg_A.size(-2))
        #     dfg_A = F.pad(dfg_A,dfg_a_0_padding)
        #     dfg_padding = dfg_A.permute(1,0)
        #     dfg_A = torch.cat((dfg_A,dfg_padding),1)
        
        #GGNN的输入要求：对于邻接矩阵，n*n的矩阵实际输入为n*2n，左半边的n*n为原始邻接矩阵，右半边的n*n矩阵为原始邻接矩阵的转置
        #GGNN对节点初始embedding的输入要求，a：原始的初始embedding，每个embedding维度为init_dim，x：padding后的节点初始embedding，每个embedding维度为hidden_dim
        CFG_nodes_x = None
        DFG_nodes_x = None
        if CFG_nodes != None:
            cfg_padding = CFG_As.permute(0,1,3,2)
            CFG_As = torch.cat((CFG_As,cfg_padding),3)

            this_device = CFG_nodes.device
            padding = torch.zeros(CFG_nodes.shape[0], CFG_nodes.shape[1], CFG_nodes.shape[2], cfg.new_config.cfg_arg.hidden_dim - CFG_nodes.shape[3]).float().to(this_device)
            CFG_nodes_x = torch.cat((CFG_nodes, padding), -1)
        
        if cg_A != None:
            cg_padding = cg_A.permute(0,2,1)
            cg_A = torch.cat((cg_A,cg_padding),2)
        
        if DFG_nodes != None:
            dfg_padding = dfg_A.permute(0,2,1)
            dfg_A = torch.cat((dfg_A,dfg_padding),2)

            this_device = DFG_nodes.device
            padding = torch.zeros(DFG_nodes.shape[0], DFG_nodes.shape[1], cfg.new_config.dfg_arg.hidden_dim - DFG_nodes.shape[2]).float().to(this_device)
            DFG_nodes_x = torch.cat((DFG_nodes, padding), -1)

            dfg_ginn_matrix_padding = dfg_ginn_matrix.permute(0,1,3,2)
            dfg_ginn_matrix = torch.cat((dfg_ginn_matrix,dfg_ginn_matrix_padding),3)
        
        result={'CFG_As':CFG_As,'CFG_nodes':CFG_nodes,'CFG_nodes_x':CFG_nodes_x,'cg_A':cg_A,'DFG_nodes':DFG_nodes,'DFG_nodes_x':DFG_nodes_x,'dfg_A':dfg_A,'dfg_ginn_matrix':dfg_ginn_matrix}

        return result #CFG_As, CFG_nodes, cg_A, DFG_nodes, dfg_A