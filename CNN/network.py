import tensorflow as tf
import math
import keras
import json
import numpy as np
from CNN.utils import KernelGen,BiasGen
# from utils import KernelGen,BiasGen
from constant import OUTPUT_DIR,CHORD_DIR
from create_templates import create_chromagram_dict
class CNN_Audio(tf.Module):
    def __init__(self,input_param,chords_list,network_map,batch_number,nodes_map,**kwargs):
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
        _,input_height,input_width,input_channels=input_param

        self.batches=batch_number
        self.input_dim=(input_height,input_width)
        self.input_channels=input_channels

        self.layers=len(network_map)
        self.network=[i[0] for i in network_map]
        self.activations=[i[1] for i in network_map]
        self.kernels_map=[i[2] for i in network_map]
        self.stride=[i[3] for i in network_map]
        self.padding=[i[4] for i in network_map]

        self.chords_count=len(chords_list)
        self.chords=chords_list

        self.data=[] #is used to store forward result in one train session

        kernelGen=KernelGen(self.kernels_map,nodes_map) #[layer,height,width,inChannels,quantity].Note: only create kernel for conv layers
        biasGen=BiasGen(self.kernels_map)#Note: only create bias for conv layers
        # if load!=None:
        #     self.b=[tf.Variable(np.load(OUTPUT_DIR+"bias_"+str(i)+load+".npy"),dtype='float32') for i in range(len(biasGen))]
        #     self.w=[tf.Variable(np.load(OUTPUT_DIR+"kernel_"+str(i)+load+".npy"),dtype='float32') for i in range(len(kernelGen))]
        # else:
        self.w=[tf.Variable(i,dtype='float32') for i in kernelGen] #[layer,height,width,inChannels,quantity]
        self.b=[tf.Variable(i,dtype='float32') for i in biasGen]
        self.learning_rate=0.001

        #used in backward
        ## for batch normalization
        self.beta=[tf.Variable(0,dtype='float32') for _ in biasGen]#offset
        self.gamma=[tf.Variable(1,dtype='float32') for _ in biasGen]#scale

        #for ema
        self.alpha=0.5 #smooth factor
        self.ema_mean=[math.nan for _ in biasGen]
        self.ema_variance=[math.nan for _ in biasGen]

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
        for i in range(0,self.layers):
            if self.network[i]=="conv":
                tmp_res=tf.nn.conv2d(tmp_res,self.w[conv_counter],self.stride[i],self.padding[i])+self.b[conv_counter]
                if is_training:
                    mean,variance=tf.nn.moments(tmp_res,[0])
                    self.ema_mean[conv_counter]=self.alpha*(mean-self.ema_mean[conv_counter])+self.ema_mean[conv_counter]
                    self.ema_variance[conv_counter]=self.alpha*(mean-self.ema_variance[conv_counter])+self.ema_variance[conv_counter]
                    tmp_res=tf.nn.batch_normalization(x=tmp_res,mean=mean,variance=variance,offset=None,scale=None,variance_epsilon=1e-6)
                else:
                    mean=self.ema_mean[conv_counter]
                    variance=self.ema_variance[conv_counter]
                tmp_res=tf.nn.batch_normalization(x=tmp_res,mean=mean,variance=variance,offset=self.beta[conv_counter],scale=self.gamma[conv_counter],variance_epsilon=1e-6)
                conv_counter+=1
                if self.activations[i]=="reLU":
                    tmp_res=tf.nn.relu(tmp_res)
                #these 2 lines are customized just for my CNN structure
                # if i<3 and is_training:
                if is_training:
                    tmp_res=tf.nn.dropout(tmp_res,0.2)
            elif self.network[i]=="pool-max":
                tmp_res=tf.nn.max_pool(tmp_res,self.kernels_map[i],self.stride[i],self.padding[i])
            elif self.network[i]=="pool-avg":
                tmp_res=tf.nn.avg_pool(tmp_res,self.kernels_map[i],self.stride[i],self.padding[i])
            elif self.network[i]=="softmax":
                tmp_res=tf.nn.softmax(tmp_res)
        #fitting
        self.data.append(tf.squeeze(tmp_res))     
    

    def backward(self,groundtruth_data,t:tf.GradientTape):
        """
        0. desc: back to the first layer once for the last data batch
        1. params:
        groundtruth_data ((batches,chords_num))
        t: gradient that this function stays within
        2. return: nothing.Update kernels and bias
        3.Note: 
        """
        cross_entropy=self.loss.__call__(groundtruth_data,self.data[-1])
        self.optimizer.minimize(cross_entropy,self.b+self.w+self.beta+self.gamma,t)
        # dw,db=t.gradient(cross_entropy,[self.w,self.b])
        # # self.optimizer.apply_gradients([(dw,self.w),(db,self.b)])
        # for i in range(len(self.w)):
        #     self.w[i].assign_sub(self.learning_rate*dw[i])
        # for i in range(len(self.b)):
        #     self.b[i].assign_sub(self.learning_rate*db[i])
    

    def evaluate(self,groundtruth_data_array):
        """
        0. desc: create confusion matrix and calculate accuracy of train result
        1. params:
        groundtruth_data_array([(batches,chords_num)])
        2. return: nothing.Update kernels and bias
        3.Note: 
        """
        # predicted_idx=tf.argmax(self.data,1)
        # confusion_matrix = tf.math.confusion_matrix(
        # groundtruth_data, predicted_idx, num_classes=self.chords_count)
        # evaluation_step = tf.reduce_mean(tf.cast(predicted_idx, tf.float32))
        # tf.summary.scalar('accuracy', evaluation_step)
        print(len(self.data))
        full_data= tf.concat(self.data,axis=0)
        # groundtruth_data_array_numpy=list(groundtruth_data_array.unbatch().as_numpy_iterator()) #for GPU
        groundtruth_data_array_numpy=tf.concat(groundtruth_data_array,axis=0) #for CPU
        predicted_value=tf.argmax(full_data,axis=1)
        groundtruth_value=tf.argmax(groundtruth_data_array_numpy,axis=1)
        # confusion_matrix=tf.math.confusion_matrix(groundtruth_value,predicted_value,self.chords_count)
        accuracy=tf.reduce_mean(tf.cast(tf.equal(predicted_value,groundtruth_value),tf.float32))
        return "Accuracy: {}%".format(accuracy*100)

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
            np.save(OUTPUT_DIR+"bias_"+str(i)+postfix,self.b[i])
            np.save(OUTPUT_DIR+"beta_"+str(i)+postfix,self.beta[i])
            np.save(OUTPUT_DIR+"gamma_"+str(i)+postfix,self.gamma[i])
            np.save(OUTPUT_DIR+"ema_mean_"+str(i)+postfix,self.ema_mean[i])
            np.save(OUTPUT_DIR+"ema_var_"+str(i)+postfix,self.ema_variance[i])

    
    #helper functions
    def toChordProb(self,chord_list,min=0.2):
        f1=open(CHORD_DIR+"chord_prob_dict.json")
        chords_prob_lst=json.load(f1).values()
        full_data= tf.concat(self.data,axis=0)
        predicted_value=full_data.numpy()/chords_prob_lst
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
        for i in range(len(self.b)):
            self.b[i]=np.load(OUTPUT_DIR+"bias_"+str(i)+postfix+".npy")
            self.w[i]=np.load(OUTPUT_DIR+"kernel_"+str(i)+postfix+".npy")
            self.beta[i]=np.load(OUTPUT_DIR+"beta_"+str(i)+postfix+".npy")
            self.gamma[i]=np.load(OUTPUT_DIR+"gamma_"+str(i)+postfix+".npy")
            self.ema_mean[i]=np.load(OUTPUT_DIR+"ema_mean_"+str(i)+postfix+".npy")
            self.ema_variance[i]=np.load(OUTPUT_DIR+"ema_var_"+str(i)+postfix+".npy")
    def clear_data(self):
        self.data=[]
    def SetLearning_rate(self,learning_rate):
        self.learning_rate=learning_rate
    def SetDataSize(self,batches):
        self.data=[0 for i in range(batches)]
