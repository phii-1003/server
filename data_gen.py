from CNN.preproccess import preprocessData,calChordProb,fixChordProb
from CNN.utils import ChordListGen
from constant import *
import numpy as np
postfix=POSTFIX_1_9_2
chord_dict=CHORD_DICT_2_CHANGED
hop_length=4410
innotation=INNOTATION_2
chord_list=ChordListGen(NOTES,innotation)
#####NOTES: CHANGE FROM CQT FEATURE TO CQT CHROMAGRAM. album 17 and 18 stops and idx 32940
#preprocess, run if there is no .npy file
##notes: from 01 to 12 is Beatles, 13 and 14 is Queen 
octaves=list(np.arange(-11/12,11/12,1/12))
preprocessData(['17','18','19'],AUDIO_DIR,CHORD_DIR,chord_dict,NOTES_DICT,chord_list,hop_length=hop_length,window_frames=19,postfix=postfix, octave_shift=octaves,usage="train")
preprocessData(['01','02','03','04','05','08','09','11','12'],AUDIO_DIR,CHORD_DIR,chord_dict,NOTES_DICT,chord_list,hop_length=hop_length,window_frames=19,postfix=postfix, octave_shift=[-9/12,-4/12,0,5/12,7/12,10/12,1],usage="valid")
# preprocessData(['06','07','12','13','14','15'],AUDIO_DIR,CHORD_DIR,chord_dict,NOTES_DICT,chord_list,hop_length=hop_length,window_frames=19,postfix=postfix, octave_shift=[-10/12,-8/12,-6/12,-4/12,-2/12,0,1/12,3/12,5/12,7/12,9/12,11/12],usage="pretrain")

    

