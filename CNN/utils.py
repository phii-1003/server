import math
import cv2
import numpy as np

#NETWORK_MAP: 
# layer type,
# activation,
# kernel shape ((height,width,inChannels, quantity) for conv layer/(height,width) for pooling layer,
# stride,
# padding

# def removeIdxInFileName(dir=CHORD_DIR):
#     """
#     0.desc: remove prefix of files in chord folder (to match the order in audio folder)
#     1.params:
#     dir(str): directory
#     2.options:
#     3.return:None
#     """
#     file_list=os.listdir(dir)
#     for i in file_list:
        
#         try:
#             int(i[:2])
#             os.rename(dir+i,dir+i[2:]) #remove the number in the file
#         except:
#             continue
def ChordListGen(notes,innotations):
    res=[]
    for i in notes:
        for j in innotations:
            res.append(i+':'+j)
    # res.append('N') #remove N chord
    return res
def BiasGen(kernel_map):
    """
    0.desc: init the biass of the network
    1.params:
    kernel_map(layer,height,width,inCHannels,quantity)
    2.options: 
    3.return: list of bias
    """
    res=[]

    for i in range(0,len(kernel_map)):
        if kernel_map[i]==None:
            continue
        if len(kernel_map[i])<4:
            continue
        else:
            res.append(np.zeros(kernel_map[i][3]))
    return res

def KernelGen(kernel_map,nodes_map):
    """
    0.desc: init the kernels of the network
    1.params:
    kernel_map(layer,height,width,inCHannels,quantity)
    nodes_map: used for kaimming gen
    2.options: 3 first kernels are Gabor, rest are Kaimming
    3.return: list of kernels
    """
    res=[]

    for i in range(0,len(kernel_map)):
        if kernel_map[i]==None:
            continue
        if i<3:
            if len(kernel_map[i])>=4:
                res.append(KernelGenHelper(kernel_map[i],option="Kaimming",information={'nodes':nodes_map[i]}))
        else:
            if len(kernel_map[i])>=4:
                res.append(KernelGenHelper(kernel_map[i],option="Kaimming",information={'nodes':nodes_map[i]}))
    return res
def KernelGenHelper(kernel_param,option="Gabor",information={}):
    """
    0.desc: init a kernel of a conv layer
    1.params:
    kernel_param(height,width,inCHannels,quantity):kernel shape
    options: either "Kaimming", "Gaussian" or "Gabor"
    information: a dict, containing extra parameter for generation
    2.options:
    3.return: kernel with shape (height,width,inChannels,quantity)
    """
    if option=="Kaimming":
        lim=np.sqrt(2/float(information["nodes"]))#nodes of previous layer
        return np.random.normal(0.0, lim, size=kernel_param)
    elif option=="Gabor": #note: gabor filter returns odd-size filter
        kernel_gabor=(kernel_param[0] if kernel_param[0]%2==1 else kernel_param[0]+1,kernel_param[1] if kernel_param[1]%2==1 else kernel_param[1]+1 )
        inChannels=kernel_param[2]
        quantity=kernel_param[3]
        steps=inChannels*quantity+1
        sigma=information["sigma"] if "sigma" in information else 2
        theta_range=information["theta_range"] if "theta_range" in information else np.arange(np.pi/steps,np.pi,np.pi/steps)
        frequency=information["lambda"] if "lambda" in information else 0.5
        phase=information["gamma"] if "gamma" in information else 0
        res=np.zeros(kernel_param)
        k=0
        for i in range(0,quantity):
            for j in range(0,inChannels):
                    res[:,:,j,i]=cv2.getGaborKernel(kernel_gabor,sigma=sigma,theta=theta_range[k],lambd=frequency,gamma=phase)[:kernel_param[0],:kernel_param[1]]
                    k+=1
        return res

def shuffle_data(input:np.ndarray,grountruth:np.ndarray):
    """
    0.desc: shuffle input and groundtruth in the same order
    1.params:
    input: input data
    groudntruth: groundtruth data
    2.options:
    3.return: input(shuffled), groundtruth(shuffled)
    """
    arr_len=input.shape[0]
    permutation=np.random.permutation(arr_len)
    return input[permutation],grountruth[permutation]
def Hz_to_MIDI(Hz):
    """
    0.desc: transfer hertz t0 note (69 equals midi note A4)
    1.params:
    Hz(float): note frequency
    2.options:
    3.return: note
    """
    return int(12*math.log2(Hz/440)) +69

def MIDI_to_Hz(note):
    """
    0.desc: transfer note to hertz (69 equals midi note A4)
    1.params:
    note(int): music note
    2.options:
    3.return: hertz
    """
    return 400*math.pow(2,(note-69)/12)

def MIDI_to_Note(midi_Note):
    """
    0.desc: transfer midi note to musical note (69 equals midi note A4)
    1.params:
    midi_Note(int): midi note
    2.options:
    3.return: musical note(str) with pitch and octave. Ex: B2, C3
    """
    pitch_array=["C","D","E","F","G","A","B"]
    octave=int((midi_Note-24)/12)+1 #24 equals C1
    pitch_number=(midi_Note-24)%12
    return str(pitch_array[pitch_number]+octave)

def Note_to_MIDI(note:str):
    """
    0.desc: transfer musical note to midi note (69 equals midi note A4)
    1.params:
    note(str): musical note with pitch and octave. Ex: B2, C3
    2.options:
    3.return: midi note(int)
    """
    #check
    if len(note)!=2:
        return -1 #fail case
    pitch_array=["C","D","E","F","G","A","B"]
    octave=int(note[1])+1
    pitch_number=pitch_array.index(note[0])
    return pitch_number+octave*12