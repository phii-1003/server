#note: these preproccessData,preprocessGroundTruth, preprocessChord functions solely made for me using the dataset in isophonics.net/content/reference-annotations-beatles 
import librosa
import numpy as np
import pandas as pd
import json
import os
def preprocessData(album_idx_list:list[str],audio_dir,chord_dir,chord_dict,notes_dict,wanted_chord_list,hop_length,window_frames,postfix):
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
    2. options: write file into Audio/data.csv and Chord/groundtruth.csv
    3. return: input_data,groundtruth_data
    """
    input_res=[]
    groundtruth_res=[]

    #calculate chord prob
    chord_prob_dict={k:0 for k in wanted_chord_list}
    total_frames=0
    ##
    for i in album_idx_list:
        audio_list=os.listdir(audio_dir+i)
        chord_list=os.listdir(chord_dir+i)
        if int(i)<16: #data from isophonic annotation. Link: http://isophonics.net/content/reference-annotations 
            if len(audio_list)!=len(chord_list):
                raise Exception('Input data is missing either audio files or chord files')
            for j,k in zip(audio_list,chord_list):
                print(j)
                audio=preprocessAudioFile(audio_dir+i+'\\'+j,window_length=window_frames,hop_length=hop_length,octave_shift=[-1,0,1])
                input_res+=audio
                chord=preprocessGroundTruthFile(chord_dir+i+'\\'+k,input_res[-1].shape[0],wanted_chord_list=wanted_chord_list,notes_dict=notes_dict,chord_dict=chord_dict,window_length=(hop_length/44100)*window_frames)
                # chord_lst=np.argmax(chord,axis=1)
                # for m in chord_lst:
                #     chord_prob_dict[wanted_chord_list[m]]+=1
                #     total_frames+=1
                # print("processed frames: ",total_frames)
                for __ in range(len(audio)):
                    groundtruth_res.append(chord)   
        else: #data from IDMT-SMT. Link: https://zenodo.org/records/7544213
            k=chord_list[0]
            for j in audio_list:
                audio=preprocessAudioFile(audio_dir+i+'\\'+j,octave_shift=[-1,0,1])
                input_res+=audio
                chord=preprocessGroundTruthFile(chord_dir+i+'\\'+k,input_res[-1].shape[0],wanted_chord_list=wanted_chord_list,notes_dict=notes_dict,chord_dict=chord_dict)
                for __ in range(len(audio)):
                    groundtruth_res.append(chord)   
    data,groundtruth= np.concatenate(input_res,axis=0),np.concatenate(groundtruth_res,axis=0)
    np.save(audio_dir+'data'+postfix,data)
    np.save(chord_dir+'groundtruth'+postfix,groundtruth)
    # with open(chord_dir+"chord_prob_dict.json", "w") as outfile: 
    #     json.dump(chord_prob_dict, outfile)
    # with open(chord_dir+"total_frames.json", "w") as outfile: 
    #     json.dump(total_frames, outfile)
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
        alter_interval=interval_list[interval]
        res.append(root+':'+alter_interval)
    return res
        
def preprocessGroundTruthFile(groundtruth,total_windows,wanted_chord_list:list[str],chord_dict,notes_dict,window_length=1.9):
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
    groundtruth_data=pd.read_csv(groundtruth,header=None,delimiter=' ')
    #for folders 13,14,15
    if groundtruth_data.shape[1]==1:
        groundtruth_data=pd.read_csv(groundtruth,header=None,delimiter='\t')
    start_time=groundtruth_data[0].to_numpy()
    end_time=groundtruth_data[1].to_numpy()
    chord=preprocessChord(groundtruth_data[2],chord_dict,notes_dict)
    if total_windows*window_length>end_time[-1]:
        start_time=np.append(start_time,end_time[-1])
        end_time=np.append(end_time,total_windows*window_length)
        chord.append('N')

    res_str=[]
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
                res_str.append(chord_appeared[duration_appeared.index(max(duration_appeared))])
                chord_appeared.clear()
                duration_appeared.clear()
                break
            else:
                idx+=1
    res_prob_list=np.zeros((len(res_str),len(wanted_chord_list)))
    for i in range(res_prob_list.shape[0]):
        idx=wanted_chord_list.index(res_str[i])
        res_prob_list[i,idx]=1
    return res_prob_list

def preprocessAudioFile(input,window_length=19,sample_rate=44100,bins_per_octave=24,total_bins=144,hop_length=2205,octave_shift=[0]):
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
        y_shifted=librosa.effects.pitch_shift(y,sr=sr,n_steps=octave*12)
        fmin = librosa.midi_to_hz(28) #min frequency that the transform hold, which is E1
        CQT = librosa.cqt(y_shifted, sr=sr, fmin=fmin, bins_per_octave=bins_per_octave,n_bins=total_bins, hop_length=hop_length)
        realNumberCQT=np.abs(CQT).astype('float32')# consider add gaussian: window=("gaussian",stdd=0.65) (stdd may change))
        if realNumberCQT.shape[1]%window_length!=0:
            realNumberCQT= np.pad(realNumberCQT,[(0,0),(0,window_length-realNumberCQT.shape[1]%window_length)])
        res.append( np.reshape(realNumberCQT,(int(realNumberCQT.shape[1]/window_length),realNumberCQT.shape[0],window_length,1)) )
    return res
    