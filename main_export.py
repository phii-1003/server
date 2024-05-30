from main_recognition import Recognition
from ExportModule.export import ExportMusic
from ExportModule.parseRecognition import parseRecognitionModuleOutput
from CNN.utils import ChordListGen
from constant import NOTES,INNOTATION_2
chord_list=ChordListGen(NOTES,INNOTATION_2)
song="audio2.mp3"
recog_obj=Recognition()
export_obj=ExportMusic(chord_list)
result_chords,window_length,duration,_=recog_obj.recognize(song)
duration=int(duration*1000)
window_length=int(window_length*1000)
json_result=parseRecognitionModuleOutput(result_chords,window_length)
export_obj.export(json_result,duration,window_length)
