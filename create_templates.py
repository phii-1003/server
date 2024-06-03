"""
Algorithm based on the paper 'Automatic Chord Recognition from
Audio Using Enhanced Pitch Class Profile' by Kyogu Lee
This script computes 12 dimensional chromagram for chord detection
@author ORCHISAMA DAS
"""

"""Create pitch profile template for 12 major and 12 minor chords and save them in a json file
Gmajor template = [1,0,0,0,1,0,0,1,0,0,0,0] - needs to be run just once"""

import json
from constant import *
def get_nested_circle_of_fifths():
    chords = [
        "N",
        "C:maj",
        "C#:maj",
        "D:maj",
        "D#:maj",
        "E:maj",
        "F:maj",
        "F#:maj",
        "G:maj",
        "G#:maj",
        "A:maj",
        "A#:maj",
        "B:maj",
        "C:min",
        "C#:min",
        "D:min",
        "D#:min",
        "E:min",
        "F:min",
        "F#:min",
        "G:min",
        "G#:min",
        "A:min",
        "A#:min",
        "B:min",
    ]
    nested_cof = [
        "C:maj",
        "E:min",
        "G:maj",
        "B:min",
        "D:maj",
        "F#:min",
        "A:maj",
        "C#:min",
        "E:maj",
        "G#:min",
        "B:maj",
        "D#:min",
        "F#:maj",
        "A#:min",
        "C#:maj",
        "F:min",
        "G#:maj",
        "C:min",
        "D#:maj",
        "G:min",
        "A#:maj",
        "D:min",
        "F:maj",
        "A:min",
    ]
    return chords, nested_cof

def create_chromagram_dict(chord_list,to_json=False,low=0.2,high=1,peak=1,neighbor_penalty=-0.25):
    """
    0. desc: create chromagrams from a list of chords
    1. param:
    chord_list: list of chords needed
    to_json: if the result saved to a JSON file
    2. options:
    3. result: a dict of chords and corresponding chromagram
    """
    chromagram_dict = dict()
    for chord in chord_list:
        chromagram=[low for _ in range(12)]
        colon=chord.find(':')
        if colon!=-1:
            note_idx=NOTES.index(chord[:colon])
            scale=CHROMAGRAM_DICT[chord[colon+1:]]
            for i in scale:
                if i==0:
                    chromagram[(note_idx+i)%12]=peak
                    chromagram[(note_idx+i+1)%12]=neighbor_penalty
                    chromagram[(note_idx+i-1)%12]=neighbor_penalty
                else:
                    chromagram[(note_idx+i)%12]=high
                    chromagram[(note_idx+i+1)%12]=neighbor_penalty
                    chromagram[(note_idx+i-1)%12]=neighbor_penalty
        chromagram_dict[chord]=chromagram

    # save as JSON file
    if to_json:
        with open("chord_templates.json", "w") as fp:
            json.dump(chromagram_dict, fp, sort_keys=False)
            print("Saved succesfully to JSON file")
    return chromagram_dict
# create_chromagram_dict()