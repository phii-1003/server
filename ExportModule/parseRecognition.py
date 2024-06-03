from constant import INNOTATION_DICT
import json
def parseRecognitionModuleOutput(chord_list:list[str],window_length,separator=':',interval_dict=INNOTATION_DICT):
    """
    0. desc: parse result of main_recognition file
    1. param:
    chord_list: list of recognized chords 
    window_length: time of each chord
    separator: char that separate root note and chord type
    2. options: JSON string format: time-chordname(E.g A:maj)- root note (E.g A) - interval (E.g major)
    3. result: list of JSON string
    """
    res_lst=[]
    idx=0
    prev_chord=("N",idx)
    for chord_name in chord_list:
        separator_idx=chord_name.find(separator)
        if separator_idx!=-1: 
            root_note=chord_name[:separator_idx]
            interval=interval_dict[chord_name[separator_idx+1:]]
        else: #N chord
            root_note=chord_name
            interval="None"
        if prev_chord[0]!=chord_name:
            tmp_dict={'Time': str(idx*window_length),"ChordName":chord_name,"ChordRoot":root_note,"ChordInterval":interval}
            json_str_res=json.dumps(tmp_dict)
            res_lst.append(json_str_res)
        prev_chord=(chord_name,idx)
        idx+=1
    return json.dumps(res_lst)

