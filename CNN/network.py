import tensorflow as tf
import math
import keras
import json
import numpy as np
from CNN.utils import KernelGen,BiasGen
# from utils import KernelGen,BiasGen
from constant import OUTPUT_DIR,CHORD_DIR
from create_templates import create_chromagram_dict
class MyLRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate,decay_step):
        self.learning_rate = initial_learning_rate
        self.decay=0.8
        self.decay_step=decay_step
    def setLR(self,lr):
        self.learning_rate=lr
    def __call__(self, step):
        return self.learning_rate*math.pow(self.decay,step/self.decay_step)
class CNN_Audio(tf.Module):
    def __init__(self,chords_list,network_map,nodes_map,**kwargs):
        """
        0. desc: init the Network parameters
        1. params:
        input_param((batches,height,width,channels))
        chords_list([]): list of wanted chords
        network_map([a1,a2,...]): see desc on utils.py. Note: quantity equals output channels
        batch_number:number of batches
        nodes_map: see calculating the input nodes of each layer
        load:load params from previous trainings
        2. return: none
        3.Note:
        network structure:
            "conv-reLU",
            "conv-reLU",
            "conv-reLU",
            "pool-max",
            "conv-reLU",
            "conv-reLU",
            "pool-max",
            "conv-reLU",
            "conv-reLU",
            "conv-reLU",
            "conv-linear",
            "pool-avg",
            "softmax"
        """
        # super.__init__(**kwargs)

        self.layers=len(network_map)
        self.network=[i[0] for i in network_map]
        self.activations=[i[1] for i in network_map]
        self.kernels_map=[i[2] for i in network_map]
        self.stride=[i[3] for i in network_map]
        self.padding=[i[4] for i in network_map]

        self.chords_count=len(chords_list)
        self.chords=chords_list

        self.data=[] #is used to store forward result in one train session

        # kernelGen=KernelGen(self.kernels_map,nodes_map) #[layer,height,width,inChannels,quantity].Note: only create kernel for conv layers
        chromagram=list(create_chromagram_dict(chords_list,to_json=False,low=0.01,high=0.8,peak=0.8,neighbor_penalty=0.01).values())
        kernelGen=[np.expand_dims(np.array(chromagram).T,axis=(1,2))]
        biasGen=BiasGen(self.kernels_map)#Note: only create bias for conv layers
        # if load!=None:
        #     self.b=[tf.Variable(np.load(OUTPUT_DIR+"bias_"+str(i)+load+".npy"),dtype='float32') for i in range(len(biasGen))]
        #     self.w=[tf.Variable(np.load(OUTPUT_DIR+"kernel_"+str(i)+load+".npy"),dtype='float32') for i in range(len(kernelGen))]
        # else:
        self.w=[tf.Variable(i,dtype='float32') for i in kernelGen] #[layer,height,width,inChannels,quantity]
        # self.b=[tf.Variable(i,dtype='float32') for i in biasGen]
        self.learning_rate=MyLRSchedule(0.001,20)

        #used in backward
        ## for batch normalization
        # self.beta=[tf.Variable(0,dtype='float32') for _ in biasGen]#offset
        # self.gamma=[tf.Variable(1,dtype='float32') for _ in biasGen]#scale
        self.beta=[tf.Variable(0,dtype='float32')]#offset
        self.gamma=[tf.Variable(1,dtype='float32')]#scale
        #for ema
        self.alpha=0.75 #smooth factor
        self.ema_mean=[math.nan for _ in self.beta]
        self.ema_variance=[math.nan for _ in self.beta]

        self.loss=keras.losses.CategoricalCrossentropy()
        self.optimizer=keras.optimizers.Adam(self.learning_rate)
    #main functions

    def forward(self,input_data,is_training=True):
        """
        0. desc: forward to the softmax layer once and append to the end of data list
        1. params:
        input_data((batches,height,width,channels))
        kernel[(height,width,inChannels,quantity)]
        2. return: store in self.data (batch,chords_num) to the corresponding idx
        3.Note: drop out for the first three layers are 0.5
        """
        tmp_res=input_data
        conv_counter=0
        def checkErr(tens):
            if tf.reduce_any(tf.math.is_nan(tens)):
                print("Nan is here")
            if tf.reduce_any(tf.math.is_inf(tens)):
                print("Inf is here")
        def calEMA(prev,x):
            return self.alpha*(prev-x)+x if x is not math.nan else prev
        for i in range(0,self.layers):
            if self.network[i]=="conv":
                tmp_res=tf.nn.conv2d(tmp_res,self.w[conv_counter],self.stride[i],self.padding[i])
                # checkErr(tmp_res)
                # checkErr(tmp_res)
                # if conv_counter==0:
                #     if is_training:
                #         mean,variance=tf.nn.moments(tmp_res,[0])
                #         self.ema_mean[0]=tf.stop_gradient(calEMA(mean,self.ema_mean[0]))
                #         self.ema_variance[0]=tf.stop_gradient(calEMA(variance,self.ema_variance[0]))
                #     else:
                #         mean=self.ema_mean[0]
                #         variance=self.ema_variance[0] #negative mean and var element causes nan. Use this in report
                #     tmp_res=tf.nn.batch_normalization(x=tmp_res,mean=mean,variance=variance,offset=self.beta[0],scale=self.gamma[0],variance_epsilon=1e-4)
                if self.activations[i]=="reLU":
                    tmp_res=tf.nn.relu(tmp_res)
                if is_training:
                    # if conv_counter<2:
                    #     tmp_res=tf.nn.dropout(tmp_res,0.25)
                    # else:
                    tmp_res=tf.nn.dropout(tmp_res,0.15)
                conv_counter+=1
            elif self.network[i]=="pool-max":
                tmp_res=tf.nn.max_pool(tmp_res,self.kernels_map[i],self.stride[i],self.padding[i])
            elif self.network[i]=="pool-avg":
                tmp_res=tf.nn.avg_pool(tmp_res,self.kernels_map[i],self.stride[i],self.padding[i])
            elif self.network[i]=="softmax":
                tmp_res=tf.nn.softmax(tmp_res)
        #fitting
        self.data.append(tf.squeeze(tmp_res))     
    
    def loss_cal(self,groundtruth_data):
        return self.loss.__call__(groundtruth_data,self.data[-1])
    def backward(self,groundtruth_data,t:tf.GradientTape):
        """
        0. desc: back to the first layer once for the last data batch
        1. params:
        groundtruth_data ((batches,chords_num))
        t: gradient that this function stays within
        2. return: nothing.Update kernels and bias
        3.Note: 
        """

        cross_entropy=self.loss_cal(groundtruth_data)
        # self.optimizer.minimize(cross_entropy,self.w+self.beta+self.gamma,t)
        self.optimizer.minimize(cross_entropy,self.w,t)
        return cross_entropy.numpy()
    

    def evaluate(self,groundtruth_data_array):
        """
        0. desc: create confusion matrix and calculate accuracy of train result
        1. params:
        groundtruth_data_array([(batches,chords_num)])
        2. return: nothing.Update kernels and bias
        3.Note: 
        """
        full_data= tf.concat(self.data,axis=0)
        groundtruth_data_array_numpy=tf.concat(groundtruth_data_array,axis=0) #for CPU
        predicted_value=tf.argmax(full_data,axis=1)
        groundtruth_value=tf.argmax(groundtruth_data_array_numpy,axis=1)
        # confusion_matrix=tf.math.confusion_matrix(groundtruth_value,predicted_value,self.chords_count)
        accuracy=tf.reduce_mean(tf.cast(tf.equal(predicted_value,groundtruth_value),tf.float32))
        return accuracy*100

    def store(self,postfix):
        """
        0. desc: store training result to files in Output dir
        1. params:
        postfix: denote chord classes and window duration
        2. return: nothing.Store weight and bias into two different files in Output
        3.Note: weight numpy array has dimension (kernel_idx,height,width,inChannels,quantity); bias has dim (bias_idx,kernelQuantity)
        """
        # bias_tensor_list=[i.read_value() for i in self.b] #for GPU
        # bias_numpy=np.array(bias_tensor_list) #for GPU
        # kernel_tensor_list=[i.read_value() for i in self.w] #for GPU
        # kernel_numpy=np.array(kernel_tensor_list) #for GPU
        for i in range(len(self.w)):
            np.save(OUTPUT_DIR+"kernel_"+str(i)+postfix,self.w[i])
            # np.save(OUTPUT_DIR+"bias_"+str(i)+postfix,self.b[i])
        for i in range(len(self.beta)):
            np.save(OUTPUT_DIR+"beta_"+str(i)+postfix,self.beta[i])
            np.save(OUTPUT_DIR+"gamma_"+str(i)+postfix,self.gamma[i])
            np.save(OUTPUT_DIR+"ema_mean_"+str(i)+postfix,self.ema_mean[i])
            np.save(OUTPUT_DIR+"ema_var_"+str(i)+postfix,self.ema_variance[i])
    
    #helper functions
    def delete_ema(self):
        self.ema_mean=[math.nan for _ in self.ema_mean]
        self.ema_variance=[math.nan for _ in self.ema_variance]
    def toChordProb(self):
        f1=open(CHORD_DIR+"chord_prob_dict.json")
        chords_prob_lst=np.array(list(json.load(f1).values()),dtype='float32')
        full_data= tf.concat(self.data,axis=0)
        predicted_value=full_data.numpy()
        
        # for i in range(len(predicted_value)):
        #     predicted_value[i,:]/=sum(predicted_value[i,:])
        return predicted_value
    def load_params(self,postfix):
        """
        0. desc: load training result from files in Output dir
        1. params:
        postfix: denote chord classes and window duration
        2. return: nothing.Store weight and bias into two different files in Output
        3.Note: weight numpy array has dimension (kernel_idx,height,width,inChannels,quantity); bias has dim (bias_idx,kernelQuantity)
        """
        #Note: must check manually if the number of kernels and bias is equal to the number of layer
        for i in range(len(self.w)):
            # self.b[i]=np.load(OUTPUT_DIR+"bias_"+str(i)+postfix+".npy")
            self.w[i]=np.load(OUTPUT_DIR+"kernel_"+str(i)+postfix+".npy")
        for i in range(len(self.beta)):
            self.beta[i]=np.load(OUTPUT_DIR+"beta_"+str(i)+postfix+".npy")
            self.gamma[i]=np.load(OUTPUT_DIR+"gamma_"+str(i)+postfix+".npy")
            self.ema_mean[i]=np.load(OUTPUT_DIR+"ema_mean_"+str(i)+postfix+".npy")
            self.ema_variance[i]=np.load(OUTPUT_DIR+"ema_var_"+str(i)+postfix+".npy")
    def clear_data(self):
        self.data=[]
    def SetLearning_rate(self,learning_rate):
        self.learning_rate.setLR(learning_rate)
    def SetDataSize(self,batches):
        self.data=[0 for i in range(batches)]
