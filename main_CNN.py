import numpy as np
from CNN.preproccess import fixChordProb,calChordProb
from CNN.train import *
from constant import *
from CNN.network import CNN_Audio
from CNN.utils import * 
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
    
    input_train=np.load(AUDIO_DIR+'train_data'+postfix+'.npy')
    groundtruth_train=np.load(CHORD_DIR+'train_groundtruth'+postfix+'.npy')
    input_pretrain=np.load(AUDIO_DIR+'pretrain_data'+postfix+'.npy')
    groundtruth_pretrain=np.load(CHORD_DIR+'pretrain_groundtruth'+postfix+'.npy')
    input_valid=np.load(AUDIO_DIR+'valid_data'+postfix+'.npy')
    groundtruth_valid=np.load(CHORD_DIR+'valid_groundtruth'+postfix+'.npy')
    #fix chord imbalance
    input_train=np.concatenate((input_train,input_pretrain))
    groundtruth_train=np.concatenate((groundtruth_train,groundtruth_pretrain))
    delete_idxs_train=fixChordProb(groundtruth_train,wanted_chord_list=chord_list,delete_amount=24000)
    input_train=np.delete(input_train,delete_idxs_train,axis=0)
    groundtruth_train=np.delete(groundtruth_train,delete_idxs_train,axis=0)
    calChordProb(groundtruth_train,wanted_chord_list=chord_list,chord_dir=CHORD_DIR)
    if option=="training":
        print("Input train size: ",input_train.shape,"Input valid size: ",input_valid.shape)
        print("Groundtruth train size: ",groundtruth_train.shape,"Groundtruth valid size: ",groundtruth_valid.shape )
        samples=input_train.shape[0]
        # divided_batch_num=math.ceil(samples/BATCH) #for GPU
        divided_batch_num=int(samples/BATCH) #for CPU

        neural_network=CNN_Audio(input_train.shape,chord_list,network_map,divided_batch_num,nodes_map)
        #pretrain
        train(neural_network,input_pretrain,groundtruth_pretrain,None,None,LEARNING_RATE_MAP_PRETRAIN)
        neural_network.delete_ema()
        #train
        train(neural_network,input_train,groundtruth_train,input_valid,groundtruth_valid,LEARNING_RATE_MAP)

        neural_network.store(postfix)
    elif option=="validating":
        samples=input_valid.shape[0]
        # divided_batch_num=math.ceil(samples/BATCH) #for GPU
        divided_batch_num=int(samples/BATCH) #for CPU
        neural_network=CNN_Audio(input_valid.shape,chord_list,network_map,divided_batch_num,nodes_map)
        neural_network.load_params(postfix)
        res=predict(neural_network,input_valid,groundtruth_valid)
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
main(option="training") 