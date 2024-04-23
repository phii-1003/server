from preproccess import preprocessData
from utils import ChordListGen
from constant import *
postfix=POSTFIX_1_9_2
chord_dict=CHORD_DICT_2
hop_length=4410
innotation=INNOTATION_2
chord_list=ChordListGen(NOTES,innotation)
#preprocess, run if there is no .npy file
##notes: from 01 to 12 is Beatles, 13 and 14 is Queen 
preprocessData(['01','02','03','04','05','06','07','08','09','11','12','13','14','15','16','17'],AUDIO_DIR,CHORD_DIR,chord_dict,NOTES_DICT,chord_list,hop_length=hop_length,window_frames=19,postfix=postfix)

