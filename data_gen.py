from CNN.preproccess import preprocessData,calChordProb
from CNN.utils import ChordListGen
from constant import *
postfix=POSTFIX_1_9_2
chord_dict=CHORD_DICT_2
hop_length=4410
innotation=INNOTATION_2
chord_list=ChordListGen(NOTES,innotation)+["N"]
#preprocess, run if there is no .npy file
##notes: from 01 to 12 is Beatles, 13 and 14 is Queen 
# preprocessData(['01','02','03','04','05','06','07','08','09','11','12','13','14','15','16','17'],AUDIO_DIR,CHORD_DIR,chord_dict,NOTES_DICT,chord_list,hop_length=hop_length,window_frames=19,postfix=postfix, octave_shift=[-1,-8/12,-2/12,0,3/12,7/12,1])
# preprocessData(['08','09','11','13','14','15'],AUDIO_DIR,CHORD_DIR,chord_dict,NOTES_DICT,chord_list,hop_length=hop_length,window_frames=19,postfix=postfix, octave_shift=[-1,-8/12,-2/12,0,3/12,7/12,1],usage="test")
# preprocessData(['01','02','03','04','05','06','07','12','16','17','18'],AUDIO_DIR,CHORD_DIR,chord_dict,NOTES_DICT,chord_list,hop_length=hop_length,window_frames=19,postfix=postfix, octave_shift=[-1,-8/12,-6/12,-2/12,0,3/12,5/12,7/12,1],usage="train")
# calChordProb(CHORD_DIR,wanted_chord_list=chord_list,postfix=postfix)
preprocessData(['08','09','11','14'],AUDIO_DIR,CHORD_DIR,chord_dict,NOTES_DICT,chord_list,hop_length=hop_length,window_frames=19,postfix=postfix, octave_shift=[-1,-2/12,0,3/12,7/12],usage="test")
preprocessData(['01','02','03','04','05','06','07','12','13','15','17','18'],AUDIO_DIR,CHORD_DIR,chord_dict,NOTES_DICT,chord_list,hop_length=hop_length,window_frames=19,postfix=postfix, octave_shift=[-8/12,-6/12,-2/12,0,3/12,5/12,7/12,1],usage="train")
calChordProb(CHORD_DIR,wanted_chord_list=chord_list,postfix=postfix)

