import tensorflow as tf
import numpy as np
from network import CNN_Audio
from sklearn.utils import shuffle
from constant import *
# # @tf.function(reduce_retracing=True)
# def train_help(neural_network:CNN_Audio,data,idx,i,learning_rate):
#     print("Epoch round {}, no.{} with learning rate {}".format(idx,i,learning_rate))
#         # #shuffle and split data
#     input_data_shuffled=data.map(lambda x,y: x)
#     groundtruth_data_shuffled=data.map(lambda x,y: y)
#     input_data_array=input_data_shuffled.batch(BATCH)
#     groundtruth_data_array=groundtruth_data_shuffled.batch(BATCH)

#     for input,groundtruth in tf.data.Dataset.zip(input_data_array,groundtruth_data_array):
#         # input=input_data_array[j]
#         # groundtruth=groundtruth_data_array[j]
#         with tf.GradientTape() as t:
#             neural_network.forward(input)
#             neural_network.backward(groundtruth,t)
#     print(neural_network.evaluate(groundtruth_data_array))
#     neural_network.clear_data()

# def train(neural_network:CNN_Audio,input_data,groundtruth_data,learning_rate_map,samples):
#     """
#     0. desc: performing train on the dataset
#     1. params:
#     neural_network: network object
#     input_data: audio data
#     groundtruth_data_array: groundtruth data
#     learning_rate_map
#     samples: number of data
#     2. options: split input and groundtruth into batches
#     3. return: [chord result]
#     """
#     data=tf.data.Dataset.from_tensor_slices((input_data,groundtruth_data))
#     data=data.shuffle(buffer_size=data.cardinality(),reshuffle_each_iteration=True)
    
#     idx=0
#     for epoch,learning_rate in learning_rate_map:
#         #find current learning rate
#         neural_network.SetLearning_rate(learning_rate)
#         for i in range(epoch):
#             train_help(neural_network,data,idx,i,neural_network.learning_rate)
            
#         idx+=1
#     # return neural_network.data

def train(neural_network:CNN_Audio,input_data,groundtruth_data,input_test,groundtruth_test,learning_rate_map,samples):
    """
    0. desc: performing train on the dataset
    1. params:
    neural_network: network object
    input_data: audio data
    groundtruth_data_array: groundtruth data
    learning_rate_map
    samples: number of data
    2. options: split input and groundtruth into batches
    3. return: [chord result]
    """
    idx=0
    for epoch,learning_rate in learning_rate_map:
        #find current learning rate
        neural_network.SetLearning_rate(learning_rate)
        for i in range(epoch):
            print("Epoch round {}, no.{} with learning rate {}".format(idx,i,learning_rate))
            input_data_shuffled,groundtruth_data_shuffled=shuffle(input_data,groundtruth_data)
            input_data_array=np.array_split(input_data_shuffled,neural_network.batches)
            groundtruth_data_array=np.array_split(groundtruth_data_shuffled,neural_network.batches)
            for j in range(len(input_data_array)):
                input=input_data_array[j]
                groundtruth=groundtruth_data_array[j]
                with tf.GradientTape() as t:
                    neural_network.forward(input)
                    neural_network.backward(groundtruth,t)
            print(neural_network.evaluate(groundtruth_data_array))
            neural_network.clear_data()
            #testing
            input_test_array=np.array_split(input_test,neural_network.batches)
            groundtruth_test_array=np.array_split(groundtruth_test,neural_network.batches)
            for input in input_test_array:
                neural_network.forward(input)
            print(neural_network.evaluate(groundtruth_test_array))
            neural_network.clear_data()
        idx+=1
    # return neural_network.data

def predict(neural_network:CNN_Audio,input,groundtruth):
    """
    0. desc: predict chords from inputs
    1. params:
    input ((batches,height,width,channels))
    groundtruth_data ([(batches,chords_num)])
    2. return: accuracy
    3.Note:
    """
    #split into batches
    samples=input.shape[0]
    divided_batch_num=int(samples/BATCH)
    input_data_array=np.array_split(input,divided_batch_num)
    groundtruth_data_array=np.array_split(groundtruth,divided_batch_num)
    for i in range(len(input_data_array)):
        neural_network.forward(input_data_array[i],False)
    return neural_network.evaluate(groundtruth_data_array)
