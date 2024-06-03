from main_recognition import Recognition
from ExportModule.export import ExportMusic
from ExportModule.parseRecognition import parseRecognitionModuleOutput
# from CNN.utils import ChordListGen
# from constant import NOTES,INNOTATION_2
# chord_list=ChordListGen(NOTES,INNOTATION_2)
# song="audio2.mp3"
# reco_obj=Recognition()
# export_obj=ExportMusic(chord_list)
# result_chords,window_length,duration,_=reco_obj.recognize(song)
# duration=int(duration*1000)
# window_length=int(window_length*1000)
# json_result=parseRecognitionModuleOutput(result_chords,window_length)
# export_obj.export(json_result,duration,window_length)

def recognize_chord(recognition_obj:Recognition,input):
    result_chords,window_length,duration,_=recognition_obj.recognize(input)
    json_result=parseRecognitionModuleOutput(result_chords,window_length)
    return json_result
def recognize_and_export(recognition_obj:Recognition,export_obj:ExportMusic,input):
    result_chords,window_length,duration,_=recognition_obj.recognize(input)
    duration=int(duration*1000)
    window_length=int(window_length*1000)
    json_result=parseRecognitionModuleOutput(result_chords,window_length)
    export_obj.export(json_result,duration,window_length)
    return json_result