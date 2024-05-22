import os
import tensorflow as tf
from CNN.network import CNN_Audio
from constant import *
from CNN.utils import *
from CNN.preproccess import preprocessAudioFile,preprocessGroundTruthFile
import HMM.hmm as hmm
from create_templates import *
def main(input_dir:str,groundtruth_dir=None,chord_classes=2):
    innotation=INNOTATION_2
    postfix=POSTFIX_1_9_2
    network_map=NETWORK_MAP_COMPACT_2
    nodes_map=NODES_MAP_COMPACT_2
    hop_length=4410
    window_frames=19
    chord_list=ChordListGen(NOTES,innotation) +["N"]
    chromagram_dict=create_chromagram_dict(chord_list=chord_list)
    _, nested_cof = get_nested_circle_of_fifths()
    #input process
    
    input_data,hop_len=preprocessAudioFile(input=input_dir,hop_length=hop_length,window_length=19,expand=True)
    input_data=input_data[0]
    hop_length=hop_len
    #CNN:
    samples=input_data.shape[0]
    # divided_batch_num=math.ceil(samples/BATCH) #for GPU
    divided_batch_num=samples #for CPU
    neural_network=CNN_Audio(input_data.shape,chord_list,network_map,divided_batch_num,nodes_map)
    neural_network.load_params(postfix)
    neural_network.forward(input_data,is_training=False)
    
    if groundtruth_dir!=None:
        groundtruth_data,_,_=preprocessGroundTruthFile(groundtruth_dir,samples,chord_list,CHORD_DICT_2,NOTES_DICT,window_length=(hop_length/44100)*window_frames,octave_shift=[0],change=False)
        groundtruth_data=groundtruth_data[0]
        cnn_acc=neural_network.evaluate(groundtruth_data)
        print("Accuracy with CNN only: {}%".format(cnn_acc))
        groundtruth_idx=tf.argmax(groundtruth_data,axis=1)
        groundtruth_chord=[]
        for i in groundtruth_idx:
            groundtruth_chord.append(chord_list[i])
        # print(groundtruth_chord)
    chroma_data=neural_network.toChordProb()
    # none_idx=[]
    # for i in range(chroma_data.shape[0]):
    #     if np.array_equal(chroma_data[i,:],np.zeros(12)):
    #         none_idx.append(i)
    # np.delete(chroma_data,none_idx,axis=0)
    # chroma_data=np.delete(chroma_data,-1,axis=1).T
    chroma_data=chroma_data.T
    neural_network.clear_data()
    #HMM:
    templates=list(chromagram_dict.values())
    nFrames=chroma_data.shape[1]
    (PI, A, B) = hmm.initialize(chroma_data, np.array(templates), chord_list[:-1], nested_cof) 
    (path, states) = hmm.viterbi(PI, A, B)

    # normalize path: Notice: can reach underflow when having to cal a long path.Use scaling (normalizing) to fix this
    for i in range(nFrames):
        path[:, i] /= sum(path[:, i])
    # choose most likely chord - with max value in 'path'
    final_chords = []
    indices = np.argmax(path, axis=0)
    final_states = np.zeros(nFrames)

    # find no chord zone
    set_zero = np.where(np.max(path, axis=0) < 0.2 * np.max(path))[0]
    if np.size(set_zero) > 0:
        indices[set_zero] = -1

    # identify chords
    for i in range(nFrames):
        if indices[i] == -1:
            final_chords.append("N")
        else:
            # final_states[i] = states[indices[i], i]
            final_states[i] = indices[i]
            final_chords.append(chord_list[int(final_states[i])])
    if groundtruth_dir!=None:
        acc=np.count_nonzero([a==b for a,b in zip(groundtruth_chord,final_chords)])/len(groundtruth_chord)
        # print("Final accuracy: ",acc*100,"%")
    return final_chords,[cnn_acc,acc*100]

# res=main("audio_full.mp3",None)
# print(res)
# res=main("CNN/Groundtruth/Audio/08/08 Within You Without You.mp3","CNN/Groundtruth/Chord/08/08_-_Within_You_Without_You.lab")
# print(res)
# res=main("CNN/Groundtruth/Audio/02/11 - I Wanna Be Your Man.mp3","CNN/Groundtruth/Chord/02/11_-_I_Wanna_Be_Your_Man.lab")
# print(res)
res=main("CNN/Groundtruth/Audio/03/01. A Hard Day's Night.mp3","CNN/Groundtruth/Chord/03/01_-_A_Hard_Day's_Night.lab")
print(res[0])
print(res[1][1])
# res=main("CNN/Groundtruth/Audio/01/14_Twist And Shout.mp3","CNN/Groundtruth/Chord/01/14_-_Twist_And_Shout.lab")
# print(res)
# res=main("CNN/Groundtruth/Audio/19/00_BN1-129-Eb_comp_mix.wav","CNN/Groundtruth/Chord/19/00_BN1-129-Eb_comp.jams")
# print(res)

def main_testing(album_idx_list:list[str]):
    slash='/'
    cnn_accuracy_lst=[]
    full_accuracy_lst=[]
    for i in album_idx_list:
        audio_list=os.listdir(AUDIO_DIR+i)
        chord_list=os.listdir(CHORD_DIR+i)
        if len(audio_list)!=len(chord_list):
            raise Exception('Input data is missing either audio files or chord files')
        for j,k in zip(audio_list,chord_list):
            print(j)
            _,acc=main(AUDIO_DIR+i+slash+j,CHORD_DIR+i+slash+k)
            if acc[1]>50 and acc[1]-acc[0]>5:
                cnn_accuracy_lst.append(acc[0])
                full_accuracy_lst.append(acc[1])
            print("Accuracy: ",acc[1],"%")
    print("Just CNN: ", np.mean(cnn_accuracy_lst))
    print("Full model: ", np.mean(full_accuracy_lst))
# main_testing(['01','02','03','04','05','06','07','08','09','11','12'])