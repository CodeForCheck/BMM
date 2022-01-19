import os
import random
from collections import defaultdict
import re
import pickle
import glob
import time
import datetime


class Node():
    def __init__(self,class_id,file_id):
        self.single=True
        self.list=[False,False,False,False]
        self.class_id=class_id
        self.file_id=file_id
        self.list_value=[]

filedic = defaultdict(Node)
key_list = None
train_key_list = set()
validation_key_list = set()
test_key_list = set()
result_validation_key_list = set()
result_test_key_list = set()
from_file = None



def load_pre_data(f_name):
    #f_name = cfg.new_config.pre_data_path +'/'+ data_type + '/' + "sample_" + str(file_i) + ".pkl"
    f = open(f_name, 'rb')
    dt = pickle.load(f)
    return dt

def get_all_file_to_list(data_list):
    #file name created: compile,file_id,class
    #file_list=os.popen('ls '+from_file)
    #for line in file_list.readlines():
    for line in data_list:
        names = re.split('/',line)
        file_tag=re.split('[_.]',names[-1])
        compile_tag=file_tag[1]
        file_id = file_tag[2]
        class_id = file_tag[3]
        key = class_id+'_'+file_id

        if key in filedic.keys():
            filedic[key].list_value.append(compile_tag)
            filedic[key].single=False
        else:
            new_node = Node(class_id,file_id)
            new_node.list_value.append(compile_tag)
            filedic[key]=new_node
    print(len(filedic))
    return filedic
   
def get_index_for_key(index,except_i): #多个编译版本中随机选择一个（除了except_i不能选）
    key = key_list[index]
    sample_len = len(filedic[key].list_value)
    sample_i = random.randint(0,sample_len-1)
    
    if except_i>=0 and sample_len==1:
        print("Error: wrong except_i")
    while sample_i == except_i:
        sample_i = random.randint(0,sample_len-1)
    return sample_i

def load_with_all_key(all_key): 
    #TODO：存储的返回值建议不再拼为字符串，而是直接返回原始的路径，这样就不需要再重新解析一遍地址了，loader.py的NewDataset_bert_pair就不用调用pair_sample.py的这个函数了
    #key_list[i]+'_'+str(except_i1)+'-'+key_list[j]+'_'+str(except_i2)
    
    pair_args = all_key.split('-')
    #print(all_key,pair_args)
    keyi,sample_i,keyj,sample_j = pair_args[0],int(pair_args[1]),pair_args[2],int(pair_args[3])
    name = from_file+'sample_'+filedic[keyi].list_value[sample_i]+'_'+filedic[keyi].file_id+'_'+filedic[keyi].class_id+'.pkl'
    sample1 = load_pre_data(name)
    sample1['key'] = keyi
    name = from_file+'sample_'+filedic[keyj].list_value[sample_j]+'_'+filedic[keyj].file_id+'_'+filedic[keyj].class_id+'.pkl'
    sample2 = load_pre_data(name)
    sample2['key'] = keyj
    return sample1, sample2

def get_key_with_all_key(all_key):
    #key_list[i]+'_'+str(except_i1)+'-'+key_list[j]+'_'+str(except_i2)
    pair_args = all_key.split('-')
    #print(all_key,pair_args)
    keyi,sample_i,keyj,sample_j = pair_args[0],int(pair_args[1]),pair_args[2],int(pair_args[3])
    return keyi, keyj


# def get_with_keyindex(index,except_i):
#     key, sample_i = get_key_and_index(index,except_i)
#     key = key_list[index]
#     #print(key,sample_i,filedic[key].list_value)
#     name = from_file+'sample_'+filedic[key].list_value[sample_i]+'_'+filedic[key].file_id+'_'+filedic[key].class_id+'.pkl'
#     sample = load_pre_data(name)
#     sample['key'] = key
#     #print('get_with_keyindex:', index, sample['key'])
#     return sample,sample_i


def create_false_pair(data_type):
    global filedic
    global key_list
    #global sample_index
    # if (data_type == 'test') or (data_type == 'validation'):
    #     min_index = 0
    #     max_index = int(len(key_list)*0.3)
    # else:
    #     min_index = int(len(key_list)*0.3)+1
    #     max_index = len(key_list)-1
    min_index = 0
    max_index = len(key_list)-1
    
    i = random.randint(min_index,max_index)
    j = random.randint(min_index,max_index)
    while i==j:
        j = random.randint(0,len(key_list)-1) #随机取两个程序，这两个程序是否仅有一个编译版本不影响
    except_i1 = get_index_for_key(i,-1)
    except_i2 = get_index_for_key(j,-1)
    all_key1 = key_list[i]+'-'+str(except_i1)+'-'+key_list[j]+'-'+str(except_i2)
    all_key2 = key_list[j]+'-'+str(except_i2)+'-'+key_list[i]+'-'+str(except_i1) 
    #print('F',all_key1,all_key2)
    if data_type == 'test':
        test_key_list.add(all_key1)
        test_key_list.add(all_key2)
        result_test_key_list.add(all_key1)
        
    elif data_type == 'validation':
        validation_key_list.add(all_key1)
        validation_key_list.add(all_key2)
        result_validation_key_list.add(all_key1)
    # elif data_type == 'train':
    #     train_key_list.add(all_key1)
    #     train_key_list.add(all_key2)
    else:
        # if (all_key1 in test_key_list) or (all_key2 in test_key_list):
        #     return None
        # if (all_key1 in validation_key_list) or (all_key2 in validation_key_list):
        #     return None
        train_key_list.add(all_key1)
    return all_key1


def create_true_pair(data_type):
    global filedic
    global key_list
    #global sample_index
    # if (data_type == 'test') or (data_type == 'validation'):
    #     min_index = 0
    #     max_index = int(len(key_list)*0.3)
    # else:
    #     min_index = int(len(key_list)*0.3)+1
    #     max_index = len(key_list)-1
    min_index = 0
    max_index = len(key_list)-1

    #i = random.randint(0,len(key_list)-1)
    i = random.randint(min_index,max_index)
    while filedic[key_list[i]].single==True: #找到一个多编译版本的程序
        #i = random.randint(0,len(key_list)-1)
        i = random.randint(min_index,max_index)
    #print("create_true_pair",i,filedic[key_list[i]].single,filedic[key_list[i]].list_value)
    except_i1 = get_index_for_key(i,-1)
    except_i2 = get_index_for_key(i,except_i1) #选择它的随机两个版本作为同源程序对
    all_key1 = key_list[i]+'-'+str(except_i1)+'-'+key_list[i]+'-'+str(except_i2)
    all_key2 = key_list[i]+'-'+str(except_i2)+'-'+key_list[i]+'-'+str(except_i1)

    #print('T',all_key1,all_key2)
    if data_type == 'test':
        test_key_list.add(all_key1)
        test_key_list.add(all_key2)
        result_test_key_list.add(all_key1)
    elif data_type == 'validation':
        validation_key_list.add(all_key1)
        validation_key_list.add(all_key2)
        result_validation_key_list.add(all_key1)
    # elif data_type == 'train':
    #     train_key_list.add(all_key1)
    #     train_key_list.add(all_key2)
    else:
        # if (all_key1 in test_key_list) or (all_key2 in test_key_list):
        #     return None
        # if (all_key1 in validation_key_list) or (all_key2 in validation_key_list):
        #     return None
        train_key_list.add(all_key1)
    return all_key1



def create_set(amount,data_type):
    index = 0
    label_0_cnt = 0
    label_1_cnt = 0
    while index < amount:
        typei = random.randint(0,9) #随机数决定这个pair是生成同源还是非同源
        if typei == 0:
            all_key = create_false_pair(data_type)
            label_0_cnt += 1
        else:
            all_key = create_true_pair(data_type)
            label_1_cnt += 1
        if all_key != None:
            index += 1
        if(index % 20000 == 0): #输出目前的数据集大小
            dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(data_type,index,dt,label_0_cnt,label_1_cnt)


    if data_type == 'test':
        #print("Data init: create_set for",data_type,",set size =",len(test_key_list))
        return list(result_test_key_list)
    elif data_type == 'validation':
        #print("Data init: create_set for",data_type,",set size =",len(validation_key_list))
        return list(result_validation_key_list)
    elif data_type == 'train':
        return list(train_key_list)




def init_print_list():
    global filedic
    global key_list
    #f_dir = "../poj_bench/pre_data/all_sample_all_type_bert32/"
    f_dir = "../poj_bench/pre_data/all_sample_all_type_bert32/" #修改为所有data_preparation_bert.py生成的程序样例存储的路径

    #指定编译类别，读取该编译类别的所有程序样例
    compiler_tag_list = "1,5" #"0,1"
    compile_class_tag_list = compiler_tag_list.split(',')        
    data = []
    for compile_class_tag in compile_class_tag_list:
        data_tmp = glob.glob(f_dir+"sample_"+compile_class_tag+"*.pkl")
        data = data + data_tmp
    print("Get all file path")

    from_file = f_dir
    filedic = get_all_file_to_list(data)
    key_list = list(filedic.keys())
    print("Dataset init:",len(key_list))

    test_amount = 20000
    train_amount = 500000

    dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(dt)
    #创建三个集合
    data_type = 'test'
    test_all_key_list = create_set(test_amount,data_type)
    data_type = 'validation'
    validation_all_key_list = create_set(test_amount,data_type)
    data_type = 'train'
    train_all_key_list = create_set(train_amount,data_type)
    print("Dataset init finish",len(test_all_key_list),len(validation_all_key_list),len(train_all_key_list))
    # for item in test_all_key_list:
    #     print(item)
    # for item in validation_all_key_list:
    #     print(item)
    # for item in train_all_key_list:
    #     print(item)
    

    #保存到文件
    data_set_save = {'filedic':filedic,'test_list':test_all_key_list,'validation_list':validation_all_key_list,'train_list':train_all_key_list}
    if compiler_tag_list == '0,1':
        f_name = 'pair_data_set_list_llvm'
    elif compiler_tag_list == '4,5':
        f_name = 'pair_data_set_list_gcc'
    elif compiler_tag_list == '1,5':
        f_name = 'pair_data_set_list_compiler_new_all'
    else:
        f_name = 'pair_data_set_list_all'

    f_save = open(f_name, 'wb')
    pickle.dump(data_set_save, f_save, pickle.HIGHEST_PROTOCOL)
    print('save to file', f_name)



# def create_false_pair(data_type):
#     #global sample_index
#     i = random.randint(0,len(key_list)-1)
#     j = random.randint(0,len(key_list)-1)
#     while i==j:
#         j = random.randint(0,len(key_list)-1)
#     sample1,except_i1 = get_with_keyindex(i,-1)
#     sample2,except_i2 = get_with_keyindex(j,-1)
#     all_key1 = key_list[i]+'_'+str(except_i1)+'-'+key_list[j]+'_'+str(except_i2)
#     all_key2 = key_list[j]+'_'+str(except_i2)+'-'+key_list[i]+'_'+str(except_i1)
#     if data_type == 'train':
#         train_key_list.add(all_key1)
#         train_key_list.add(all_key2)
#     else:
#         if (all_key1 in train_key_list) or (all_key2 in train_key_list):
#             return None, None
#     return sample1, sample2
#     # fname1 = to_file+data_type+'/sample'+str(sample_index)+'_a'
#     # fname2 = to_file+data_type+'/sample'+str(sample_index)+'_b'
#     # f1 = open(fname1 + '.pkl', 'wb')
#     # f2 = open(fname2 + '.pkl', 'wb')
#     # sample_index += 1
#     # pickle.dump(sample1, f1, pickle.HIGHEST_PROTOCOL)
#     # pickle.dump(sample2, f2, pickle.HIGHEST_PROTOCOL)

# def create_true_pair(data_type):
#     #global sample_index
#     i = random.randint(0,len(key_list)-1)
#     while filedic[key_list[i]].single==True:
#         i = random.randint(0,len(key_list)-1)
#     #print("create_true_pair",i,filedic[key_list[i]].single,filedic[key_list[i]].list_value)
#     sample1,except_i1 = get_with_keyindex(i,-1)
#     sample2,except_i2 = get_with_keyindex(i,except_i1)
#     all_key1 = key_list[i]+'_'+str(except_i1)+'-'+key_list[i]+'_'+str(except_i2)
#     all_key2 = key_list[i]+'_'+str(except_i2)+'-'+key_list[i]+'_'+str(except_i1)
#     #print(all_key1,all_key2)
#     if data_type == 'train':
#         train_key_list.add(all_key1)
#         train_key_list.add(all_key2)
#     else:
#         if (all_key1 in train_key_list) or (all_key2 in train_key_list):
#             return None, None
#     return sample1,sample2
#     # fname1 = to_file+data_type+'/sample'+str(sample_index)+'_a.pkl'
#     # fname2 = to_file+data_type+'/sample'+str(sample_index)+'_b.pkl'
#     # f1 = open(fname1, 'wb')
#     # f2 = open(fname2, 'wb')
#     # sample_index += 1
#     # pickle.dump(sample1, f1, pickle.HIGHEST_PROTOCOL)
#     # pickle.dump(sample2, f2, pickle.HIGHEST_PROTOCOL)



# if __name__ == '__main__':
#     get_all_file_to_list()
#     key_list = list(filedic.keys())
#     #print(filedic)

#     train_amount= 20 #30000
#     validation_amount= 4 #5000
#     test_amount= 4 #5000
#     data_type='train'
#     for i in range(train_amount):
#         typei = random.randint(0,1)
#         if typei % 2 == 0:
#             create_false_pair(data_type)
#         else:
#             create_true_pair(data_type)
#     data_type='validation'  
#     sample_index = 0  
#     for i in range(validation_amount):
#         typei = random.randint(0,1)
#         if typei % 2 == 0:
#             create_false_pair(data_type)
#         else:
#             create_true_pair(data_type)
#     data_type='test'
#     sample_index = 0  
#     for i in range(test_amount):
#         typei = random.randint(0,1)
#         if typei % 2 == 0:
#             create_false_pair(data_type)
#         else:
#             create_true_pair(data_type)
    

if __name__ == '__main__': #TODO：未使用，没从这里调用，可以修改为从main调用，而不是从train.py调用
    from_file = "../poj_bench/pre_data/all_sample_all_type_bert32/"
    f_name = 'pair_data_set_list_compiler_new_all'
    f = open(f_name, 'rb')
    dt = pickle.load(f)
    test_list = dt['test_list']
    validation_list = dt['validation_list']
    train_list = dt['train_list']
    filedic = dt['filedic']

    all_key_list = train_list
    new_train_list = []
    label_cnt = 0
    label_1_cnt = 0
    label_0_cnt = 0
    for index in range(len(all_key_list)): #len(all_key_list)
        key_a, key_b = get_key_with_all_key(all_key_list[index])
        label_cnt += 1
        if key_a == key_b:
            label_1_cnt += 1
            new_train_list.append(all_key_list[index])
        else:
            label_0_cnt += 1
            new_train_list.append(all_key_list[index])
    print('label cnt (0/1/all)',label_0_cnt, label_1_cnt, label_cnt)
    
    # data_set_save = {'filedic':filedic,'test_list':test_list,'validation_list':validation_list,'train_list':new_train_list}

    # f_name = 'pair_data_set_list_compiler_new'
    # f_save = open(f_name, 'wb')
    # pickle.dump(data_set_save, f_save, pickle.HIGHEST_PROTOCOL)
    # print('save to file', f_name, label_0_cnt, label_1_cnt, label_cnt)


