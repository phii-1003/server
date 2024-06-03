from main_recognition import Recognition
from main_export import recognize_and_export,recognize_chord
from ExportModule.export import ExportMusic
from CNN.utils import ChordListGen
from constant import NOTES,INNOTATION_2
import json
import sys
chord_list=ChordListGen(NOTES,INNOTATION_2)
reco_obj=Recognition()
export_obj=ExportMusic(chord_list)
sys.stdout.write("Ready")
sys.stdout.flush()
def main():
    while True:
        inp=input()
        parsed_input=json.loads(inp)
        willExport=True if parsed_input["willExport"]=="true" else False
        if willExport:
            res=recognize_and_export(reco_obj,export_obj,"input_data.mp3")
            # sys.stdin.flush()
            # sys.stdout.flush()
            # print(res,flush=True)
            sys.stdout.write('\n')
            sys.stdout.write(res+'\n')
            sys.stdout.flush()
        else:
            res=recognize_chord(reco_obj,"input_data.mp3")
            # sys.stdin.flush()
            # sys.stdout.flush()
            # print(res,flush=True)
            sys.stdout.write('\n')
            sys.stdout.write(res+'\n')
            sys.stdout.flush()
        sys.stdin.flush()
main()