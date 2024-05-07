import librosa
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
    chord_list=ChordListGen(NOTES,innotation) +["N"]
    chromagram_dict=create_chromagram_dict(chord_list=chord_list)
    _, nested_cof = get_nested_circle_of_fifths()
    #input process
    
    input_data=preprocessAudioFile(input=input_dir,hop_length=hop_length)[0]

    #CNN:
    samples=input_data.shape[0]
    # divided_batch_num=math.ceil(samples/BATCH) #for GPU
    divided_batch_num=int(samples/BATCH) #for CPU
    neural_network=CNN_Audio(input_data.shape,chord_list,network_map,divided_batch_num,nodes_map)
    neural_network.load_params(postfix)
    neural_network.forward(input_data,False)
    
    if groundtruth_dir!=None:
        groundtruth_data,_,_=preprocessGroundTruthFile(groundtruth_dir,samples,chord_list,CHORD_DICT_2,NOTES_DICT,window_length=1.9,octave_shift=[0],change=False)
        groundtruth_data=groundtruth_data[0]
        print("Accuracy with CNN only: {}%".format(neural_network.evaluate(groundtruth_data)))
        groundtruth_idx=tf.argmax(groundtruth_data,axis=1)
        groundtruth_chord=[]
        for i in groundtruth_idx:
            groundtruth_chord.append(chord_list[i])
        print(groundtruth_chord)
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
    nFrames=samples
    (PI, A, B) = hmm.initialize(chroma_data, np.array(templates), chord_list[:-1], nested_cof) 
    (path, states) = hmm.viterbi(PI, A, B)

    # normalize path
    for i in range(nFrames):
        path[:, i] /= sum(path[:, i])

    # choose most likely chord - with max value in 'path'
    final_chords = []
    indices = np.argmax(path, axis=0)
    final_states = np.zeros(nFrames)

    # find no chord zone
    set_zero = np.where(np.max(path, axis=0) < 0.3 * np.max(path))[0]
    if np.size(set_zero) > 0:
        indices[set_zero] = -1

    # identify chords
    for i in range(nFrames):
        if indices[i] == -1:
            final_chords.append("N")
        else:
            final_states[i] = states[indices[i], i]
            final_chords.append(chord_list[int(final_states[i])])
    # for i in none_idx:
    #     final_chords.insert(i,"N")
    acc=np.count_nonzero([a==b for a,b in zip(groundtruth_chord,final_chords)])/len(groundtruth_chord)
    print("Final accuracy: ",acc*100,"%")
    return final_chords

res=main("08 Within You Without You.mp3","08_-_Within_You_Without_You.lab")
print(res)
res=main("09. You Never Give Me Your Money.mp3","09_-_You_Never_Give_Me_Your_Money.lab")
print(res)
res=main("02_Misery.mp3","02_-_Misery.lab")
print(res)
###B is currently 12-D, Change B to 25-D? Or replace mutirivate gaussian to bayes prob for init B (remove N chord prob)