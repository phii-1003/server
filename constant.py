#############################CNN TRAIN####################################
BATCH=240

POSTFIX_0_95_2="_0_95_2"
POSTFIX_1_9_2="_1_9_2"
POSTFIX_0_95_5="_0_95_5"
POSTFIX_1_9_5="_1_9_5"
POSTFIX_0_95_7="_0_95_7"
POSTFIX_1_9_7="_1_9_7"

# LEARNING_RATE_MAP=[
#     (4,0.05),
#     (4,0.03),
#     (4,0.02),
#     (4,0.01)
# ]
LEARNING_RATE_MAP_PRETRAIN=[
    (4,6e-4),
]

LEARNING_RATE_MAP=[
    (10,5e-4),
    (5,4e-4),
]

NODES_MAP_BIG_2=[
144*19,
32*142*17,
32*140*15,
32*140*15,
32*70*15,
64*68*13,
64*66*11,
64*33*11,
128*13*3,
128*13*3,
64*13*3,
25*13*3,
25*1*1]
NETWORK_MAP_BIG_2=[
    ["conv","reLU",(3,3,1,32),(1,1),"VALID"],
    ["conv","reLU",(3,3,32,32),(1,1),"VALID"],
    ["conv","reLU",(3,3,32,32),(1,1),"SAME"],
    ["pool-max",None,(2,1),(2,1),"VALID"],
    ["conv","reLU",(3,3,32,64),(1,1),"VALID"],
    ["conv","reLU",(3,3,64,64),(1,1),"VALID"],
    ["pool-max",None,(2,1),(2,1),"VALID"],
    ["conv","reLU",(21,9,64,128),(1,1),"VALID"],
    ["conv","reLU",(3,3,128,128),(1,1),"SAME"],
    ["conv","reLU",(3,3,128,64),(1,1),"SAME"],
    ["conv","linear",(1,1,64,25),(1,1),"VALID"],
    ["pool-avg",None,(13,3),(1,1),"VALID"],
    ["softmax",None,None,None,None]
]

NODES_MAP_BIG_5=[
144*19,
32*142*17,
32*140*15,
32*140*15,
32*70*15,
64*68*13,
64*66*11,
64*33*11,
128*13*3,
128*13*3,
64*13*3,
61*13*3,
61*1*1]
NETWORK_MAP_BIG_5=[
    ["conv","reLU",(3,3,1,32),(1,1),"VALID"],
    ["conv","reLU",(3,3,32,32),(1,1),"VALID"],
    ["conv","reLU",(3,3,32,32),(1,1),"SAME"],
    ["pool-max",None,(2,1),(2,1),"VALID"],
    ["conv","reLU",(3,3,32,64),(1,1),"VALID"],
    ["conv","reLU",(3,3,64,64),(1,1),"VALID"],
    ["pool-max",None,(2,1),(2,1),"VALID"],
    ["conv","reLU",(21,9,64,128),(1,1),"VALID"],
    ["conv","reLU",(3,3,128,128),(1,1),"SAME"],
    ["conv","reLU",(3,3,128,64),(1,1),"SAME"],
    ["conv","linear",(1,1,64,61),(1,1),"VALID"],
    ["pool-avg",None,(13,3),(1,1),"VALID"],
    ["softmax",None,None,None,None]
]
NODES_MAP_BIG_7=[
144*19,
32*142*17,
32*140*15,
32*140*15,
32*70*15,
64*68*13,
64*66*11,
64*33*11,
128*13*3,
128*13*3,
64*13*3,
85*13*3,
85*1*1]
NETWORK_MAP_BIG_7=[
    ["conv","reLU",(3,3,1,32),(1,1),"VALID"],
    ["conv","reLU",(3,3,32,32),(1,1),"VALID"],
    ["conv","reLU",(3,3,32,32),(1,1),"SAME"],
    ["pool-max",None,(2,1),(2,1),"VALID"],
    ["conv","reLU",(3,3,32,64),(1,1),"VALID"],
    ["conv","reLU",(3,3,64,64),(1,1),"VALID"],
    ["pool-max",None,(2,1),(2,1),"VALID"],
    ["conv","reLU",(21,9,64,128),(1,1),"VALID"],
    ["conv","reLU",(3,3,128,128),(1,1),"SAME"],
    ["conv","reLU",(3,3,128,64),(1,1),"SAME"],
    ["conv","linear",(1,1,64,85),(1,1),"VALID"],
    ["pool-avg",None,(13,3),(1,1),"VALID"],
    ["softmax",None,None,None,None]
]

# NODES_MAP_COMPACT_2=[
# 12*19,
# 36*10*17,
# 36*8*15,
# 36*4*5,
# 24*4*5,
# 24*1*1]
# NETWORK_MAP_COMPACT_2=[
#     ["conv","reLU",(3,3,1,120),(1,1),"VALID"],
#     ["conv","reLU",(3,3,120,120),(1,1),"VALID"],
#     ["pool-max",None,(2,3),(2,3),"VALID"],
#     ["conv","linear",(1,1,120,24),(1,1),"VALID"],
#     ["pool-avg",None,(4,5),(1,1),"VALID"],
#     ["softmax",None,None,None,None]
# ]

# NODES_MAP_COMPACT_2=[
# 144*19,
# 32*141*16,
# 32*138*13,
# 32*135*10,
# 32*27*10,
# 64*13*3,
# 64*13*3,
# 24*13*3,
# 24*1*1]
# NETWORK_MAP_COMPACT_2=[
#     ["conv","reLU",(4,4,1,32),(1,1),"VALID"],
#     ["conv","reLU",(4,4,32,32),(1,1),"VALID"],
#     ["conv","reLU",(4,4,32,48),(1,1),"VALID"],
#     ["pool-max",None,(5,1),(5,1),"VALID"],
#     ["conv","reLU",(15,8,48,48),(1,1),"VALID"],
#     ["conv","reLU",(4,4,48,48),(1,1),"SAME"],
#     ["conv","linear",(1,1,48,24),(1,1),"VALID"],
#     ["pool-avg",None,(13,3),(1,1),"VALID"],
#     ["softmax",None,None,None,None]
# ]

NODES_MAP_COMPACT_2=[
12*19,
24*12*5,
24*10*3,
24*8*1,
24*1*1]
NETWORK_MAP_COMPACT_2=[
    ["pool-avg",None,(1,7),(1,3),"VALID"],
    ["conv","reLU",(3,3,1,24),(1,1),"SAME"],
    ["conv","reLU",(3,3,24,4),(1,1),"VALID"],
    ["conv","reLU",(3,3,4,24),(1,1),"VALID"],
    ["pool-avg",None,(8,1),(1,1),"VALID"],
    ["softmax",None,None,None,None]
]

NODES_MAP_COMPACT_5=[
144*19,
32*141*16,
32*138*13,
32*46*13,
64*13*3,
64*13*3,
61*13*3,
61*1*1]
NETWORK_MAP_COMPACT_5=[
    ["conv","reLU",(4,4,1,32),(1,1),"VALID"],
    ["conv","reLU",(4,4,32,32),(1,1),"VALID"],
    ["pool-max",None,(3,1),(3,1),"VALID"],
    ["conv","reLU",(34,11,32,64),(1,1),"VALID"],
    ["conv","reLU",(4,4,64,64),(1,1),"SAME"],
    ["conv","linear",(1,1,64,61),(1,1),"VALID"],
    ["pool-avg",None,(13,3),(1,1),"VALID"],
    ["softmax",None,None,None,None]
]

NODES_MAP_COMPACT_7=[
144*19,
32*141*16,
32*138*13,
32*46*13,
64*13*3,
64*13*3,
85*13*3,
85*1*1]
NETWORK_MAP_COMPACT_7=[
    ["conv","reLU",(4,4,1,32),(1,1),"VALID"],
    ["conv","reLU",(4,4,32,32),(1,1),"VALID"],
    ["pool-max",None,(3,1),(3,1),"VALID"],
    ["conv","reLU",(34,11,32,64),(1,1),"VALID"],
    ["conv","reLU",(4,4,64,64),(1,1),"SAME"],
    ["conv","linear",(1,1,64,85),(1,1),"VALID"],
    ["pool-avg",None,(13,3),(1,1),"VALID"],
    ["softmax",None,None,None,None]
]
NOTES=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

INNOTATION_2=['maj','min']
INNOTATION_5=['maj','min','maj7','min7','7']
INNOTATION_7=['maj','min','maj7','min7','7','dim','sus4']

CHORD_DICT_2={'':'maj','maj':'maj','min':'min','dim':'min','aug':'maj','maj7':'maj','min7':'min','7':'maj','dim7':'min','hdim7':'min','minmaj7':'maj','maj6':'maj','min6':'min','9':'maj','maj9':'maj','min9':'min','sus4':'maj','sus2':'min','5':'min'}
CHORD_DICT_2_CHANGED={'':'maj','maj':'maj','min':'min','dim':'X','aug':'X','maj7':'maj','min7':'min','7':'maj','dim7':'X','hdim7':'X','minmaj7':'min','maj6':'maj','min6':'min','9':'X','maj9':'maj','min9':'min','sus4':'X','sus2':'X','5':'X'}
CHORD_DICT_5={'':'maj','maj':'maj','min':'min','dim':'min7','aug':'7','maj7':'maj7','min7':'min7','7':'7','dim7':'min7','hdim7':'min7','minmaj7':'maj7','maj6':'maj7','min6':'min7','9':'7','maj9':'maj7','min9':'min7','sus4':'maj','sus2':'min','5':'min'}
CHORD_DICT_7={'':'maj','maj':'maj','min':'min','dim':'dim','aug':'7','maj7':'maj7','min7':'min7','7':'7','dim7':'dim','hdim7':'dim','minmaj7':'maj7','maj6':'maj7','min6':'min7','9':'7','maj9':'maj7','min9':'min7','sus4':'sus4','sus2':'min','5':'min'}

NOTES_DICT={'Bb':'A#','Ab':'G#','Gb':'F#','Eb':'D#','Db':'C#','Cb':'B','Fb':'E'}

AUDIO_DIR="/mnt/e/Local code/Official work/CNN/Groundtruth/Audio/"
CHORD_DIR="/mnt/e/Local code/Official work/CNN/Groundtruth/Chord/"
OUTPUT_DIR="/mnt/e/Local code/Official work/CNN/Output/"

# AUDIO_DIR="CNN\\Groundtruth\\Audio\\"
# CHORD_DIR="CNN\\Groundtruth\\Chord\\"
# OUTPUT_DIR="CNN\\Output\\"

######################COMMON######################
CHROMAGRAM_DICT={'maj':[0,4,7],'min':[0,3,7],'maj7':[0,4,7,11],'min7':[0,3,7,10],'7':[0,4,7,10],'dim':[0,3,6],'sus4':[0,5,7]}