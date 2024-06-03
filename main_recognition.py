import os
import tensorflow as tf
from CNN.network import CNN_Audio
from constant import *
from CNN.utils import *
from CNN.preproccess import preprocessAudioFile,preprocessGroundTruthFile
import HMM.hmm as hmm
from create_templates import *
class Recognition():
    def __init__(self):
        #init constants
        innotation=INNOTATION_2
        postfix=POSTFIX_1_9_2
        network_map=NETWORK_MAP_COMPACT_2
        nodes_map=NODES_MAP_COMPACT_2
        hop_length=4410
        window_frames=19
        chord_list=ChordListGen(NOTES,innotation) +["N"]
        chromagram_dict=create_chromagram_dict(chord_list=chord_list)
        _, nested_cof = get_nested_circle_of_fifths()
        self.innotation=innotation
        self.postfix=postfix
        self.network_map=network_map
        self.hop_length=hop_length
        self.window_frames=window_frames
        self.chord_list=chord_list
        self.chromagram_dict=chromagram_dict
        self.nested_cof=nested_cof

        #init CNN
        self.neural_network=CNN_Audio(chord_list,network_map,nodes_map)
        self.neural_network.load_params(postfix)
    def recognize(self,input_dir:str,groundtruth_dir=None,chord_classes=2):
        """
        return: final_chords,window_length,duration,accuracy(if possible)
        """
        #input process
        input_data,hop_len,duration=preprocessAudioFile(input=input_dir,hop_length=self.hop_length,window_length=19,expand=True)
        input_data=input_data[0]
        hop_length=hop_len

        #CNN:
        samples=input_data.shape[0]
        self.neural_network.forward(input_data,is_training=False)
        cnn_acc=None
        if groundtruth_dir!=None:
            groundtruth_data,_,_=preprocessGroundTruthFile(groundtruth_dir,samples,self.chord_list,CHORD_DICT_2,NOTES_DICT,window_length=(hop_length/44100)*self.window_frames,octave_shift=[0],change=False)
            groundtruth_data=groundtruth_data[0]
            cnn_acc=self.neural_network.evaluate(groundtruth_data)
            print("Accuracy with CNN only: {}%".format(cnn_acc))
            groundtruth_idx=tf.argmax(groundtruth_data,axis=1)
            groundtruth_chord=[]
            for i in groundtruth_idx:
                groundtruth_chord.append(self.chord_list[i])

        chroma_data=self.neural_network.toChordProb()
        chroma_data=chroma_data.T
        self.neural_network.clear_data()

        #HMM:
        templates=list(self.chromagram_dict.values())
        nFrames=chroma_data.shape[1]
        (PI, A, B) = hmm.initialize(chroma_data, np.array(templates), self.chord_list[:-1], self.nested_cof) 
        (path, states) = hmm.viterbi(PI, A, B)

        for i in range(nFrames):
            path[:, i] /= sum(path[:, i])
        final_chords = []
        indices = np.argmax(path, axis=0)
        final_states = np.zeros(nFrames)

        # find no chord zone
        set_zero = np.where(np.max(path, axis=0) < 0.4 * np.max(path))[0]
        if np.size(set_zero) > 0:
            indices[set_zero] = -1

        # identify chords
        for i in range(nFrames):
            if indices[i] == -1:
                final_chords.append("N")
            else:
                # final_states[i] = states[indices[i], i]
                final_states[i] = indices[i]
                final_chords.append(self.chord_list[int(final_states[i])])

        acc=None
        if groundtruth_dir!=None:
            acc=np.count_nonzero([a==b for a,b in zip(groundtruth_chord,final_chords)])/len(groundtruth_chord)
            acc*=100
            # print("Final accuracy: ",acc*100,"%")
        return final_chords,(hop_length/44100)*self.window_frames,duration,[cnn_acc,acc]
    
# def main(input_dir:str,groundtruth_dir=None,chord_classes=2):
#     innotation=INNOTATION_2
#     postfix=POSTFIX_1_9_2
#     network_map=NETWORK_MAP_COMPACT_2
#     nodes_map=NODES_MAP_COMPACT_2
#     hop_length=4410
#     window_frames=19
#     chord_list=ChordListGen(NOTES,innotation) +["N"]
#     chromagram_dict=create_chromagram_dict(chord_list=chord_list)
#     _, nested_cof = get_nested_circle_of_fifths()
#     #input process
    
#     input_data,hop_len,_=preprocessAudioFile(input=input_dir,hop_length=hop_length,window_length=19,expand=True)
#     input_data=input_data[0]
#     hop_length=hop_len
#     #CNN:
#     samples=input_data.shape[0]
#     neural_network=CNN_Audio(chord_list,network_map,nodes_map)
#     neural_network.load_params(postfix)
#     neural_network.forward(input_data,is_training=False)
#     cnn_acc=None
#     if groundtruth_dir!=None:
#         groundtruth_data,_,_=preprocessGroundTruthFile(groundtruth_dir,samples,chord_list,CHORD_DICT_2,NOTES_DICT,window_length=(hop_length/44100)*window_frames,octave_shift=[0],change=False)
#         groundtruth_data=groundtruth_data[0]
#         cnn_acc=neural_network.evaluate(groundtruth_data)
#         print("Accuracy with CNN only: {}%".format(cnn_acc))
#         groundtruth_idx=tf.argmax(groundtruth_data,axis=1)
#         groundtruth_chord=[]
#         for i in groundtruth_idx:
#             groundtruth_chord.append(chord_list[i])
#         # print(groundtruth_chord)
#     chroma_data=neural_network.toChordProb()
#     print(np.argmax(chroma_data,axis=1))
#     # none_idx=[]
#     # for i in range(chroma_data.shape[0]):
#     #     if np.array_equal(chroma_data[i,:],np.zeros(12)):
#     #         none_idx.append(i)
#     # np.delete(chroma_data,none_idx,axis=0)
#     # chroma_data=np.delete(chroma_data,-1,axis=1).T
#     chroma_data=chroma_data.T
#     neural_network.clear_data()
#     #HMM:
#     templates=list(chromagram_dict.values())
#     nFrames=chroma_data.shape[1]
#     (PI, A, B) = hmm.initialize(chroma_data, np.array(templates), chord_list[:-1], nested_cof) 
#     (path, states) = hmm.viterbi(PI, A, B)

#     # normalize path: Notice: can reach underflow when having to cal a long path.Use scaling (normalizing) to fix this
#     for i in range(nFrames):
#         path[:, i] /= sum(path[:, i])
#     # choose most likely chord - with max value in 'path'
#     final_chords = []
#     indices = np.argmax(path, axis=0)
#     final_states = np.zeros(nFrames)

#     # find no chord zone
#     set_zero = np.where(np.max(path, axis=0) < 0.4 * np.max(path))[0]
#     if np.size(set_zero) > 0:
#         indices[set_zero] = -1

#     # identify chords
#     for i in range(nFrames):
#         if indices[i] == -1:
#             final_chords.append("N")
#         else:
#             # final_states[i] = states[indices[i], i]
#             final_states[i] = indices[i]
#             final_chords.append(chord_list[int(final_states[i])])
#     acc=None
#     if groundtruth_dir!=None:
#         acc=np.count_nonzero([a==b for a,b in zip(groundtruth_chord,final_chords)])/len(groundtruth_chord)
#         acc*=100
#         # print("Final accuracy: ",acc*100,"%")
#     return final_chords,[cnn_acc,acc]
# res=main("audio2.mp3",None)
# print(res[0])
# print(res[1][1])
# res=main("CNN/Groundtruth/Audio/08/08 Within You Without You.mp3","CNN/Groundtruth/Chord/08/08_-_Within_You_Without_You.lab")
# print(res)
# res=main("CNN/Groundtruth/Audio/02/11 - I Wanna Be Your Man.mp3","CNN/Groundtruth/Chord/02/11_-_I_Wanna_Be_Your_Man.lab")
# print(res)
# res=main("CNN/Groundtruth/Audio/03/01. A Hard Day's Night.mp3","CNN/Groundtruth/Chord/03/01_-_A_Hard_Day's_Night.lab")
# print(res[0])
# print(res[1][1])
# res=main("CNN/Groundtruth/Audio/01/14_Twist And Shout.mp3","CNN/Groundtruth/Chord/01/14_-_Twist_And_Shout.lab")
# print(res)
# res=main("CNN/Groundtruth/Audio/19/00_BN1-129-Eb_comp_mix.wav","CNN/Groundtruth/Chord/19/00_BN1-129-Eb_comp.jams")
# print(res)



