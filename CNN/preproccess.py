#note: these preproccessData,preprocessGroundTruth, preprocessChord functions solely made for me using the dataset in isophonics.net/content/reference-annotations-beatles 
import librosa
import numpy as np
import pandas as pd
import json
import os
import jams
import math
import scipy.ndimage
def calChordProb(groundtruth_train,wanted_chord_list,chord_dir):
    chord_prob_dict={k:0 for k in wanted_chord_list}
    total_frames=groundtruth_train.shape[0]
    groundtruth_train_value=np.argmax(groundtruth_train,axis=1)
    for val in groundtruth_train_value:
        chord_prob_dict[wanted_chord_list[val]]+=1
    #save chord prob
    chord_prob_dict={k:v/total_frames for k,v in chord_prob_dict.items()}
    with open(chord_dir+"chord_prob_dict.json", "w") as outfile: 
        json.dump(chord_prob_dict, outfile)
    print(total_frames)
    print(chord_prob_dict)
def fixChordProb(groundtruth,wanted_chord_list,amount=25000,delete=True):
    #aim: fix number of major chords
    alter_chord_idxs=[]
    chord_alter_idxs=[]
    ismajor_int=0 if delete else 1
    chord_alter_count={wanted_chord_list[i]:0 for i in range(len(wanted_chord_list)) if i%2==ismajor_int}
    chord_prob_dict={k:0 for k in wanted_chord_list}
    total_frames=groundtruth.shape[0]
    groundtruth_train_value=np.argmax(groundtruth,axis=1)
        
    for i in range(len(groundtruth_train_value)):
        val=groundtruth_train_value[i]
        chord_prob_dict[wanted_chord_list[val]]+=1
        if len(alter_chord_idxs)>=amount:
            break
        if val%2==ismajor_int: #major if 0, minor if 1
            # if major_chord_delete_count[wanted_chord_list[val]]<delete_amount/10:
            #     if prev_val!=val:
            #         prev_val=val
            #         delete_chord_idxs.append(i)
            #         chord_prob_dict[wanted_chord_list[val]]-=1
            #         major_chord_delete_count[wanted_chord_list[val]]+=1
            #         total_frames-=1
            #     else:
            #         prev_val=-1
            chord_alter_idxs.append(i)
    while len(alter_chord_idxs)<amount:
        i=np.random.randint(0,len(chord_alter_idxs))
        val=chord_alter_idxs[i]
        if val not in alter_chord_idxs:
            if chord_alter_count[wanted_chord_list[groundtruth_train_value[val]]]>=amount/12:
                continue
            alter_chord_idxs.append(val)
            chord_prob_dict[wanted_chord_list[groundtruth_train_value[val]]]-=1
            chord_alter_count[wanted_chord_list[groundtruth_train_value[val]]]+=1
            total_frames-=1
    print(total_frames)
    print(chord_prob_dict)
    return alter_chord_idxs

def preprocessData(album_idx_list:list[str],audio_dir,chord_dir,chord_dict,notes_dict,wanted_chord_list,hop_length,window_frames,postfix,octave_shift=[-1,0,1],usage="test"):
    """
    0. desc: preproccess both audio file and their groundtruth
    1. params:
    album_list_idx [str]: list of idx of chosen album to training and predicting
    audio_dir: dir of audio data file
    chord_dir: dir of chord data file
    chord_dict: dict of chords in raw data and coressponding chords for network
    notes_dict: dict of notes and corresponding note (Ex: Bb->A#)
    wanted_chord_list: list of wanted chords
    hop_length: samples between two frames
    window_frames:number of features in a windows
    postfix: indicator for file saving
    octave_shift: data augmentation
    usage: for testing or training
    2. options: write file into Audio/data.csv and Chord/groundtruth.csv; dupplicate no chord zone
    3. return: input_data,groundtruth_data
    """
    slash='\\'
    DUPLICATE_NUM=20
    SAMPLES_COUNT=0
    input_res=[]
    groundtruth_res=[]
    ##
    for i in album_idx_list:
        audio_list=os.listdir(audio_dir+i)
        chord_list=os.listdir(chord_dir+i)
        if int(i)<16: #data from isophonic annotation. Link: http://isophonics.net/content/reference-annotations
            if len(audio_list)!=len(chord_list):
                raise Exception('Input data is missing either audio files or chord files')
            for j,k in zip(audio_list,chord_list):
                print(j)
                audio=preprocessAudioFile(audio_dir+i+slash+j,window_length=window_frames,hop_length=hop_length,octave_shift=octave_shift)
                chord,remove_idxs,padding_dict=preprocessGroundTruthFile(chord_dir+i+slash+k,audio[0].shape[0],wanted_chord_list=wanted_chord_list,notes_dict=notes_dict,chord_dict=chord_dict,window_length=(hop_length/44100)*window_frames,octave_shift=octave_shift,change=True)

                # if bool(padding_dict): #pad audio
                #     for idx,idx_lst in padding_dict.items():
                #         start,end=idx_lst
                #         for  audio_idx in range(len(audio)):
                #             if start!=0:
                #                 audio[audio_idx][idx,:,:start,:]=1/12
                #             if end!=0:
                #                 audio[audio_idx][idx,:,-1*end:,:]=1/12

                if len(remove_idxs)!=0: #remove audio
                    for audio_idx in range(len(audio)):
                        audio[audio_idx]=np.delete(audio[audio_idx],remove_idxs,0)
                groundtruth_res+=chord
                input_res+=audio
                SAMPLES_COUNT+=input_res[-1].shape[0]*len(octave_shift)
                print("Samples processed: ",SAMPLES_COUNT)
                if audio[-1].shape[0]!=groundtruth_res[-1].shape[0] or len(chord)!=len(audio):
                    print("Error in this file")
                    print("Shape: ",audio[-1].shape,groundtruth_res[-1].shape)
                    print("Shifts: ",len(chord),len(audio))
                else:
                    print("All clear")
        elif int(i)>15 and int(i)<17: #silence sounds in 16th folder (from freesound.com)
            if len(audio_list)!=len(chord_list):
                raise Exception('Input data is missing either audio files or chord files')
            for j,k in zip(audio_list,chord_list):
                print(j)
                audio=preprocessAudioFile(audio_dir+i+slash+j,window_length=window_frames,hop_length=hop_length,octave_shift=[-6/12,-2/12,0,3/12,5/12,7/12])
                chord,remove_idxs,padding_dict=preprocessGroundTruthFile(chord_dir+i+slash+k,audio[0].shape[0],wanted_chord_list=wanted_chord_list,notes_dict=notes_dict,chord_dict=chord_dict,window_length=(hop_length/44100)*window_frames,octave_shift=[-6/12,-2/12,0,3/12,5/12,7/12],change=True)

                # if bool(padding_dict): #pad audio
                #     for idx,idx_lst in padding_dict.items():
                #         start,end=idx_lst
                #         for  audio_idx in range(len(audio)):
                #             if start!=0:
                #                 audio[audio_idx][idx,:,:start,:]=1e-6
                #             if end!=0:
                #                 audio[audio_idx][idx,:,-1*end:,:]=1e-6

                if len(remove_idxs)!=0: #remove audio
                    for audio_idx in range(len(audio)):
                        audio[audio_idx]=np.delete(audio[audio_idx],remove_idxs,0)
                groundtruth_res+=chord
                input_res+=audio
                SAMPLES_COUNT+=input_res[-1].shape[0]*6
                print("Samples processed: ",SAMPLES_COUNT)
                if audio[-1].shape[0]!=groundtruth_res[-1].shape[0] or len(chord)!=len(audio):
                    print("Error in this file")
                    print("Shape: ",audio[-1].shape,groundtruth_res[-1].shape)
                    print("Shifts: ",len(chord),len(audio))
                else:
                    print("All clear")
        elif int(i)<=18 and int(i)>=17: #data from IDMT-SMT. Link: https://zenodo.org/records/7544213
            octave_shift_2=[-1,-3/4,-1/2,0,3/4,1,2]
            k=chord_list[0]
            for j in audio_list:
                print(j)

                audio=preprocessAudioFile(audio_dir+i+slash+j,window_length=math.ceil(window_frames/10)*10,hop_length=hop_length,octave_shift=octave_shift_2)
                for l in range(len(audio)):
                    audio[l]=audio[l][:,:,:-1,:]
                chord,remove_idxs,padding_dict=preprocessGroundTruthFile(chord_dir+i+slash+k,audio[0].shape[0],window_length=math.ceil((hop_length/44100)*window_frames),wanted_chord_list=wanted_chord_list,notes_dict=notes_dict,chord_dict=chord_dict,octave_shift=octave_shift_2,change=True)
                # if bool(padding_dict): #pad audio
                #     for idx,idx_lst in padding_dict.items():
                #         start,end=idx_lst
                #         for  audio_idx in range(len(audio)):
                #             if start!=0:
                #                 audio[audio_idx][idx,:,:start,:]=1e-6
                #             if end!=0:
                #                 audio[audio_idx][idx,:,-1*end:,:]=1e-6

                if len(remove_idxs)!=0: #remove audio
                    for audio_idx in range(len(audio)):
                        audio[audio_idx]=np.delete(audio[audio_idx],remove_idxs,0)
                groundtruth_res+=chord
                input_res+=audio
                SAMPLES_COUNT+=input_res[-1].shape[0]*len(octave_shift_2)#length of octave shift for folder 17 and 18
                print("Samples processed: ",SAMPLES_COUNT) 
                if audio[-1].shape[0]!=groundtruth_res[-1].shape[0] or len(chord)!=len(audio):
                    print("Error in this file")
                    print("Shape: ",audio[-1].shape,groundtruth_res[-1].shape)
                    print("Shifts: ",len(chord),len(audio))
                else:
                    print("All clear")
        else: #data from GuitarSet. Link: https://zenodo.org/records/3371780
            if len(audio_list)!=len(chord_list):
                raise Exception('Input data is missing either audio files or chord files')
            for j,k in zip(audio_list,chord_list):
                print(j)
                audio=preprocessAudioFile(audio_dir+i+slash+j,window_length=window_frames,hop_length=hop_length,octave_shift=octave_shift)
                chord,remove_idxs,padding_dict=preprocessGroundTruthFile(chord_dir+i+slash+k,audio[0].shape[0],wanted_chord_list=wanted_chord_list,notes_dict=notes_dict,chord_dict=chord_dict,window_length=(hop_length/44100)*window_frames,octave_shift=octave_shift,change=True)

                # if bool(padding_dict): #pad audio
                #     for idx,idx_lst in padding_dict.items():
                #         start,end=idx_lst
                #         for  audio_idx in range(len(audio)):
                #             if start!=0:
                #                 audio[audio_idx][idx,:,:start,:]=1e-6
                #             if end!=0:
                #                 audio[audio_idx][idx,:,-1*end:,:]=1e-6

                if len(remove_idxs)!=0: #remove audio
                    for audio_idx in range(len(audio)):
                        audio[audio_idx]=np.delete(audio[audio_idx],remove_idxs,0)
                groundtruth_res+=chord
                input_res+=audio
                SAMPLES_COUNT+=input_res[-1].shape[0]*len(octave_shift)
                print("Samples processed: ",SAMPLES_COUNT)
                if audio[-1].shape[0]!=groundtruth_res[-1].shape[0] or len(chord)!=len(audio):
                    print("Error in this file")
                    print("Shape: ",audio[-1].shape,groundtruth_res[-1].shape)
                    print("Shifts: ",len(chord),len(audio))
                else:
                    print("All clear")
    data,groundtruth= np.concatenate(input_res,axis=0),np.concatenate(groundtruth_res,axis=0)
    np.save(audio_dir+usage+"_"+'data'+postfix,data)
    np.save(chord_dir+usage+"_"+'groundtruth'+postfix,groundtruth)
def preprocessChord(source_list:list[str],interval_list:dict,notes_dict):
    """
    0. desc: preproccess chords appearing in the dataset to desired chords
    1. params:
    source_list: list of chords that needs processing
    interval_dict: list of possible intervals that the dataset holds and alternative chord
    notes_dict: dict of notes and corresponding note (Ex: Bb->A#)
    2. options: 
    3. return: fixed chord (str)
    """
    res=[]
    for i in source_list:
        root=""
        interval=""
        if i=='N':
            res.append(i)
            continue
        colon_idx=i.find(':')
        slash_idx=i.find('/')
        paren_idx=i.find('(')
        if colon_idx==-1:
            if slash_idx==-1:
                root=i
                interval=":maj"
            else:
                root=i[:slash_idx]
                interval=':maj'
            if root in notes_dict:
                root=notes_dict[root]
            res.append(root+interval)
            continue

        root=i[:colon_idx]
        if root in notes_dict:
            root=notes_dict[root]

        if slash_idx==-1 or paren_idx==-1:
            end_interval_idx=max(slash_idx,paren_idx)
        else:
            end_interval_idx=min(slash_idx,paren_idx)
        if end_interval_idx==-1:
            interval=i[colon_idx+1:]
        else:
            interval=i[colon_idx+1:end_interval_idx]
            #bass=i[bass_idx+1:]
        if interval in interval_list.keys():
            alter_interval=interval_list[interval]
        else:
            alter_interval='X'
        if alter_interval!='X':
            res.append(root+':'+alter_interval)
        else:
            res.append(alter_interval)
    return res
        
def preprocessGroundTruthFile(groundtruth:str,total_windows,wanted_chord_list:list[str],chord_dict,notes_dict,window_length=1.9,octave_shift=[-1,0,1],change=False):
    """
    0. desc: preproccess groundtruth data to chunks of chord in each windows (multiple frames combine)
    1. params:
    groundtruth(.csv/.lab): file dir containing correct chord and their time
    total_windows(int): nummber of windows in the song
    wanted_chord_list: list of wanted chords
    window_length (float): duration (s) of each window
    chord_dict(dict): dict of existing chords in the groundtruth and their coressponding chord for usage in network
    notes_dict: dict of notes and corresponding note (Ex: Bb->A#)
    2. options: exist a value called No_Chord
    3. return: np.array([chord propability list] (61*1 or 85*1))
    """
    #check file extension
    extension=groundtruth[groundtruth.rfind('.')+1:]
    if extension=="lab":
        groundtruth_data=pd.read_csv(groundtruth,header=None,delimiter=' ')
        #for folders 13,14,15
        if groundtruth_data.shape[1]==1:
            groundtruth_data=pd.read_csv(groundtruth,header=None,delimiter='\t')
        start_time=groundtruth_data[0].to_numpy()
        end_time=groundtruth_data[1].to_numpy()
        chord=preprocessChord(groundtruth_data[2],chord_dict,notes_dict)
    elif extension=="jams":
        groundtruth_data=jams.load(groundtruth)
        groundtruth_data=groundtruth_data.search(namespace='chord')[-1]
        groundtruth_data=groundtruth_data.to_interval_values()
        start_time=groundtruth_data[0][:,0]
        end_time=groundtruth_data[0][:,1]
        chord=preprocessChord(groundtruth_data[1],chord_dict,notes_dict)
    if total_windows*window_length>end_time[-1]:
        start_time=np.append(start_time,end_time[-1])
        end_time=np.append(end_time,total_windows*window_length)
        chord.append('N')
    res_str=[]
    remove_idxs=[]
    padding_dict={}
    idx=0 # idx of the window, starting at time windows_idx*window_length and ending at time (windows_idx+1)*window_length
    #use for loop
    chord_appeared=[]
    duration_appeared=[]
    for i in range(total_windows):
        while True:
            chord_appeared.append(chord[idx])
            overlap=max(0,min(end_time[idx],(i+1)*window_length)-max(start_time[idx],i*window_length))
            duration_appeared.append(overlap)
            if end_time[idx]>=(i+1)*window_length:
                if change:
                    if len(chord_appeared)==1: # adding window
                        if chord_appeared[0]=="N" or chord_appeared[0]=="X": 
                            remove_idxs.append(i)#remove N chord and X chord (X means unidentified)
                            # res_str.append(chord_appeared[0]) #add N chord
                        else:
                            res_str.append(chord_appeared[0])
                    else:
                        max_duration=max(duration_appeared)
                        if max_duration>=window_length*0.85: #padding windows
                            max_duration_idx=duration_appeared.index(max_duration)
                            if chord_appeared[max_duration_idx]=="N" or chord_appeared[max_duration_idx]=="X":
                                remove_idxs.append(i)#remove N chord
                                # res_str.append(chord_appeared[max_duration_idx]) #add N chord
                                # padding_dict[i]=[int(sum(duration_appeared[:max_duration_idx]*10)),0 if max_duration_idx==len(duration_appeared)-1 else int(sum(duration_appeared[max_duration_idx:]*10))] #add N chord
                            else:
                                res_str.append(chord_appeared[max_duration_idx])
                                padding_dict[i]=[int(sum(duration_appeared[:max_duration_idx]*10)),0 if max_duration_idx==len(duration_appeared)-1 else int(sum(duration_appeared[max_duration_idx:]*10))]
                        else: #removing window
                            remove_idxs.append(i)
                else:
                    res_str.append(chord_appeared[duration_appeared.index(max(duration_appeared))])
                chord_appeared.clear()
                duration_appeared.clear()
                if end_time[idx]==(i+1)*window_length:
                    idx+=1
                break
            else:   
                idx+=1
    res_prob_list=[np.zeros((len(res_str),len(wanted_chord_list)))for _ in range(len(octave_shift))]
    shifted_chord_list=wanted_chord_list if len(wanted_chord_list)%12==0 else wanted_chord_list[:-1] #remove "N chord" for shifting
    innotation_num=len(shifted_chord_list)/12
    for shift_idx in range(len(octave_shift)):
        for res_str_idx in range(len(res_str)):
            idx=wanted_chord_list.index(res_str[res_str_idx])
            if idx==len(shifted_chord_list) or isinstance(octave_shift[shift_idx],int):
                idx=idx
            else:
                idx=int((shifted_chord_list.index(res_str[res_str_idx])+int(innotation_num*12*octave_shift[shift_idx]))%24)
            res_prob_list[shift_idx][res_str_idx][idx]=1
            
    
    # for i in range(res_prob_list.shape[0]):
    #     idx=wanted_chord_list.index(res_str[i])
    #     res_prob_list[i,idx]=1
    return res_prob_list,remove_idxs,padding_dict

def preprocessAudioFile(input,window_length=19,sample_rate=44100,bins_per_octave=24,total_bins=144,hop_length=2205,octave_shift=[0],expand=False):
    """
    0. desc: preproccess input data - audio file - to output - mel spectrogram
    1. params:
    input(.mp4,.wav,...): audio file dir
    window_length (int): number of frames in each window
    octave_shift: list of octave shifting 
    2. options: cast type to float32
    CQT;each frame last 0.1 seconds based on hop_length 4410 and sr 44100 
    3. return: [data(batches,height,width,channels=1)]
    4. Note: needs to install ffmpeg because certain file (.mp3) causes PySoundFile to fail and have to use audioread
    """
    res=[]
    y, sr = librosa.load(input, sr=sample_rate)
    for octave in octave_shift:
        y_shifted=librosa.effects.pitch_shift(y,sr=sr,n_steps=octave*12) if octave!=0 else y
        fmin = librosa.midi_to_hz(28) #min frequency that the transform hold, which is E1
        y_harmonic=librosa.effects.harmonic(y=y_shifted,margin=8)
        chroma_CQT_harmonic = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, fmin=fmin, bins_per_octave=bins_per_octave,n_octaves=int(total_bins/bins_per_octave), hop_length=hop_length)
        chroma_CQT_filter = np.minimum(chroma_CQT_harmonic,librosa.decompose.nn_filter(chroma_CQT_harmonic,aggregate=np.median,metric='cosine'))
        chroma_CQT_smooth = scipy.ndimage.median_filter(chroma_CQT_filter, size=(1, 9))
        if chroma_CQT_smooth.shape[1]%window_length!=0:
            chroma_CQT_smooth= np.pad(chroma_CQT_smooth,[(0,0),(0,window_length-chroma_CQT_smooth.shape[1]%window_length)])
        samples=int(chroma_CQT_smooth.shape[1]/window_length)
        append_res=[]
        if expand:
            for sample in range(samples):
                tmp_res=np.concatenate([np.expand_dims(np.max(chroma_CQT_smooth[:,samp:min(samp+3,samples*window_length)],axis=1),axis=1) for samp in range(sample*19,sample*19+19)],axis=1)
                append_res.append(tmp_res)
            append_res=np.array(append_res)
            append_res=np.expand_dims(append_res,axis=(3))
        else:
            append_res=np.array(np.split(chroma_CQT_smooth,samples,axis=1))
            append_res=np.expand_dims(append_res,axis=(3))
        res.append(append_res)
    return res
    