from main_recognition import Recognition
from main_export import recognize_and_export,recognize_chord
from ExportModule.export import ExportMusic
from CNN.utils import ChordListGen
from constant import NOTES,INNOTATION_2
import json
import sys
chord_list=ChordListGen(NOTES,INNOTATION_2)
recog_obj=Recognition()
export_obj=ExportMusic(chord_list)

def main():
    while True:
        inp=input("")
        parsed_input=json.loads(inp)
        willExport=parsed_input["willExport"]
        if willExport:
            res=recognize_and_export(recog_obj,export_obj,"input_data.mp3")
            print(res)
        else:
            res=recognize_chord(recog_obj,"input_data.mp3")
            print(res)
        sys.stdin.flush()
main()