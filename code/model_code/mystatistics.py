import torch
import numpy as np
import cfg


class Statistics():
    def __init__(self,task,subtask):
        self.task = task
        self.subtask = subtask
        self.accuracy = 0
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.prediction_right = 0
        self.prediction_all = 0

    def clear_cnt(self):
        self.accuracy = 0
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.prediction_right = 0
        self.prediction_all = 0

    def add_cnt(self,prediction,label,run_type):
        if self.task == 'deadstore':          
            for i in range(len(label)):
                p = prediction[i].index(max(prediction[i]))
                t = label[i].index(max(label[i]))
                train_correct01 = ((p==0)&(t==1))
                train_correct10 = ((p==1)&(t==0))
                train_correct11 = ((p==1)&(t==1))
                train_correct00 = ((p==0)&(t==0))

                self.FN += int(train_correct01)       
                self.FP += int(train_correct10)
                self.TP += int(train_correct11)
                self.TN += int(train_correct00)

        if self.task == 'binaryClassify' and self.subtask == 'binaryPair': 
            t = label           
            for i in range(len(t)):
                if run_type == 'test':
                    #print('prediction,label',prediction[i],t[i])
                    print('p,l',format(prediction[i],'.2f'),int(t[i][0])) #输出预测距离值，真实label
                #TODO：实际上binary相似性检测任务时，下面的TP TN FP FN没有意义，因为prediction也就是距离，不是以0.5为界，它没有明确的界限
                train_correct01 = ((prediction[i]<=0.5)&(t[i]==1))
                train_correct10 = ((prediction[i]>0.5)&(t[i]==0))
                train_correct11 = ((prediction[i]>0.5)&(t[i]==1))
                train_correct00 = ((prediction[i]<=0.5)&(t[i]==0))

                self.FN += int(train_correct01)       
                self.FP += int(train_correct10)
                self.TP += int(train_correct11)
                self.TN += int(train_correct00)

                #print("TP TN FP FN:",self.TP,self.TN,self.FP,self.FN)
                
        
        else:
            for i in range(len(label)):
                self.prediction_all += 1
                if run_type == 'test':
                    print("maxValue, predictClass, realClass:",max(prediction[i]),prediction[i].index(max(prediction[i])),label[i].index(max(label[i]))) #最大值，最大值对应的类别，真实类别
                if label[i][prediction[i].index(max(prediction[i]))] == 1:
                    self.prediction_right += 1

               
    def print_epoch_reuslt(self,epoch_loss,run_type):
        #task: funcClassify,binaryClassify,bug,deadstore
        if (self.task == 'deadstore') or (self.task=='binaryClassify' and self.subtask=='binaryPair'):
            #print(run_type,"TP TN FP FN ",self.TP,self.TN,self.FP,self.FN)
            if self.TP==0:
                accuracy = (self.TP+self.TN) / (self.TP+self.FP+self.TN+self.FN)
                print(run_type,"precise recall accuracy TP TN FP FN:",0,0,accuracy,self.TP,self.TN,self.FP,self.FN)
                R=0
                P=0
            else:
                P = self.TP/ (self.TP+self.FP)
                R = self.TP/ (self.TP+self.FN)
                accuracy = (self.TP+self.TN) / (self.TP+self.FP+self.TN+self.FN)
                print(run_type,"precise recall accuracy TP TN FP FN:",P,R,accuracy,self.TP,self.TN,self.FP,self.FN)
        else:
            accuracy = self.prediction_right/self.prediction_all
            print(run_type,"Accuracy prediction_right prediction_all ",accuracy,self.prediction_right,self.prediction_all)
          
        # if accuracy > cfg.new_config.accuracy_max:
        #     cfg.new_config.accuracy_max = accuracy
        #     cfg.new_config.accuracy_stop_raise=0
        # else:
        #     cfg.new_config.accuracy_stop_raise += 1
        #     print("max_accuracy, stop_raise_cnt:",cfg.new_config.accuracy_max,cfg.new_config.accuracy_stop_raise)
        #     if cfg.new_config.accuracy_stop_raise > 20:
        #         print("stop: accuracy raise stop.")
        #         return True

        if run_type == 'validation': #确认loss是否还在持续降低，如果长时间不降，就不再训练了
            if epoch_loss < cfg.new_config.validation_loss_min:
                cfg.new_config.validation_loss_min = epoch_loss
                cfg.new_config.valid_loss_stop_down = 0
            else:
                cfg.new_config.valid_loss_stop_down += 1
                print("min_loss,stop decrease cnt:",cfg.new_config.validation_loss_min,cfg.new_config.valid_loss_stop_down)
                if cfg.new_config.valid_loss_stop_down > 20:
                    print("stop: loss decrease stop.")
                    return True
        return False



