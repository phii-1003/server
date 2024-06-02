import json
import math
import os
from pydub import AudioSegment,silence,playback
from constant import SIMUL_PATH,SEQUENCE_PATH
class ExportMusic():
    def __init__(self,chord_list):
        chord_audio_lst_simul={}
        chord_audio_lst_sequence={}
        #add audio segments 
        audio_simul=os.listdir(SIMUL_PATH)
        audio_sequence=os.listdir(SEQUENCE_PATH)

        for i in audio_simul:
            tmp_simul=AudioSegment.from_file(SIMUL_PATH+i)
            file_name=i[:i.index('.')]
            file_name=file_name.replace(' ',':')
            if file_name not in chord_list:
                raise NameError("Wrong chord name")
            chord_audio_lst_simul[file_name]=tmp_simul

        for i in audio_sequence:
            tmp_sequence=AudioSegment.from_file(SEQUENCE_PATH+i)
            file_name=i[:i.index('.')]
            file_name=file_name.replace(' ',':')
            if file_name not in chord_list:
                raise NameError("Wrong chord name")
            chord_audio_lst_sequence[file_name]=tmp_sequence

        self.chord_audio_lst_simul=chord_audio_lst_simul
        self.chord_audio_lst_sequence=chord_audio_lst_sequence

    def export(self,json_lst,duration,window_length,vocal_data=AudioSegment|None):
        """
        0. desc: generate music based on found chords
        1. param:
        json_lst: TimeStamp object
        duration(ms): song duration
        window_length(ms): duration of every window of chord detecting module. It is already calculated on preprocessAudio so that every uiwndow frame is equal 4 beat so the task is just to ring the chord on that time.
        bpm: beat per minute
        signature: time signature
        2. options: assume that time signature is 4; yet to use sequence 
        3. result: audio saved in tmp.mp3
        """
        canvas=AudioSegment.empty()
        chord_sequence=[]
        time_stamp_sequence=[]
        type_sequence=[] #( is simul chord (True/False),cut frame)
        json_lst=json.loads(json_lst)
        json_list=[json.loads(i) for i in json_lst]
        #parse json list and divide into beats
        for i in range(len(json_lst)):
            chord_name=json_list[i]["ChordName"]
            start_time=int(float(json_list[i]["Time"]))
            if i==len(json_list)-1:
                end_time=duration
            else:
                end_time=int(float(json_list[i+1]["Time"]))
            chord_duration=end_time-start_time
            chord_span=math.ceil(chord_duration/window_length)
            if chord_name=="N":
                chord_sequence.append(chord_name)
                time_stamp_sequence.append(duration)
                type_sequence.append(True,0)
            chord_audio_simul=self.chord_audio_lst_simul[chord_name]
            chord_audio_sequence=self.chord_audio_lst_sequence[chord_name]
            for i in range(chord_span):
                # if i%2==0:
                #     if chord_duration-i*window_length< 400: #at least 0.4s for a simul audio chord to complete:
                #         continue
                    chord_sequence.append(chord_name)
                    time_stamp_sequence.append(start_time+i*window_length)
                    type_sequence.append((True,min(chord_duration-i*window_length,len(chord_audio_simul))))
        
        #add to canvas based on beats
        for chord_name,time_stamp,audio_type in zip(chord_sequence,time_stamp_sequence,type_sequence):
            if chord_name=="N":
                # silence_chord=AudioSegment.silent(time_stamp)
                # canvas+=silence_chord
                continue
            chord_audio=self.chord_audio_lst_simul[chord_name] if audio_type[0] else self.chord_audio_lst_sequence[chord_name]
            #pad with silence chord
            silence_chord=AudioSegment.silent(duration=time_stamp-len(canvas))
            canvas+=silence_chord
            canvas+=chord_audio[:audio_type[1]]
            # playback.play(canvas)
        #saving to tmp
        canvas.export('tmp.mp3')
