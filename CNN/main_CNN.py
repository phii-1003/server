import numpy as np
from train import *
from constant import *
from network import CNN_Audio
from utils import * 
from sklearn.model_selection import train_test_split
#uncomment lines that states for GPU if needed
def main(option="training",innotation=INNOTATION_2,postfix=POSTFIX_1_9_2,network_map=NETWORK_MAP_COMPACT_2,nodes_map=NODES_MAP_COMPACT_2):
    """
    0. desc: run graph
    1. params:
    option ("training" or "predicting"): indicate whether main will run training operation or predicting operation on the dataset
    ##other params are predeclared and change manually
    2. return: accuracy
    3.Note:
    """
    chord_list=ChordListGen(NOTES,innotation)
    
    input_data=np.load(AUDIO_DIR+'data'+postfix+'.npy')
    groundtruth_data=np.load(CHORD_DIR+'groundtruth'+postfix+'.npy')
    print("Groundtruth and input data size: ",groundtruth_data.shape,input_data.shape)
    if option=="training":
        #divide training and testing

        input_train,input_test,groundtruth_train,groundtruth_test=train_test_split(input_data,groundtruth_data,test_size=0.2)

        print("Input train size: ",input_train.shape,"Input test size: ",input_test.shape)
        print("Groundtruth train size: ",groundtruth_train.shape,"Groundtruth test size: ",groundtruth_test.shape )
        samples=input_train.shape[0]
        # divided_batch_num=math.ceil(samples/BATCH) #for GPU
        divided_batch_num=int(samples/BATCH) #for CPU

        neural_network=CNN_Audio(input_train.shape,chord_list,network_map,divided_batch_num,nodes_map)
        train(neural_network,input_train,groundtruth_train,LEARNING_RATE_MAP,samples)
        #testing
        input_test_array=np.array_split(input_test,neural_network.batches)
        groundtruth_test_array=np.array_split(groundtruth_test,neural_network.batches)
        for input in input_test_array:
            neural_network.forward(input)
        print(neural_network.evaluate(groundtruth_test_array))
        neural_network.clear_data()
        neural_network.store(postfix)
    elif option=="validating":
        samples=input_data.shape[0]
        # divided_batch_num=math.ceil(samples/BATCH) #for GPU
        divided_batch_num=int(samples/BATCH) #for CPU
        neural_network=CNN_Audio(input_data.shape,chord_list,network_map,divided_batch_num,nodes_map)
        neural_network.load_b_and_w(postfix)
        res=predict(neural_network,input_data,groundtruth_data)
        print(res)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
# Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=3500)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
main(option="validating") 