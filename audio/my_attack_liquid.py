import speech_recognition as sr
import hydra
import sys
sys.path.append(r'/liquid-s4')
sys.path.append(r'/great-score')
from audio_utils import SequenceLightningModule,SpeechCommandsDataset,my_model

from transforms import FixAudioLength, LoadAudio, ToMelSpectrogram, ToTensor
from os import path
import os.path
from ssa_core import ssa, ssa_predict, ssaview, inv_ssa, ssa_cutoff_order
from os import system
from os import listdir
from os.path import isfile, join
import wave
import scipy as sc
import librosa
import IPython.display as ipd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
import librosa as lb
# from sklearn.preprocessing import normalize
import scipy
from sklearn.decomposition import PCA
import pandas as pd
from os import listdir
from os.path import isfile, join
import time
from itertools import product
import datetime
import sys
import io
import argparse
# NOTE: Should we remove SSA or tell them how to get it?



import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from tqdm.auto import tqdm

import src.models.nn.utils as U
import src.utils as utils
import src.utils.train
from src.dataloaders import SequenceDataset  # TODO make registry
from src.tasks import decoders, encoders, tasks
from src.utils import registry
from src.utils.optim.ema import build_ema_optimizer
from src.utils.optim_groups import add_optimizer_hooks

log = src.utils.train.get_logger(__name__)

# Turn on TensorFloat32 (speeds up large model training substantially)
import torch.backends
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


import os



import argparse
import os

import torch
import numpy as np

from functools import reduce
from natsort import natsorted
from scipy import linalg
from scipy.stats import norm, entropy
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from tqdm.auto import tqdm

import sys

sys.path.append(r'/state-spaces/sashimi/sc09_classifier')


CLASSES = 'zero, one, two, three, four, five, six, seven, eight, nine'.split(', ')


from transforms import FixAudioLength, LoadAudio, ToMelSpectrogram, ToTensor

#Parsing commandline input flags
parser = argparse.ArgumentParser()
parser.add_argument("-ifile", "--inputfile", help="input audio file location")
parser.add_argument("-ofile","--outputfile",help="output file with full file location")
parser.add_argument("-a","--attack",help="Attack type, either ssa or fft")
# parser.add_argument("","",help=)
# parser.add_argument("","",help=)

#List of comparison terms
df_attack = 'Attack'
df_attack_factor = 'Attack Factor'
df_avg_diff_frame = 'Avg Diff Frame'
df_excited = 'Excited'
df_file_name = 'File Name'
df_gsm = 'GSM'
df_gender = 'Gender'
df_l2 = 'L2'
df_model_name = 'Model Name'
df_og_label = 'OG Label'
df_og_word = 'OG Word'
df_perturb_window = 'Perturb Window'
df_phoneme = 'Phoneme'
df_raster_width = 'Raster Width'
df_succ = 'Succ'
df_temporal = 'Temporal'
df_transcribed_word = 'Transcribed Word'
df_zero_locs = 'Zero Locs'

# Attacks
floor_atk_name = 'floor'
dct_atk_name = 'dct'
fft_atk_name = 'fft'
dct_base_atk_name = 'dct_base'
svd_atk_name = 'svd'
ssa_atk_name = 'ssa'
pca_atk_name = 'pca'
sin_atk_name = 'sin'

# Models 
google_phone = 'google_phone'
wit = 'wit'
sphinx = 'sphinx'
deepspeech = 'deepspeech'
google = 'google'

liquid_s4='liquid-s4'
state_space='state-space'



#Defenses
quantization = "quantization"




def normalize(data):
    normalized = data.ravel()*1.0/np.amax(np.abs(data.ravel()))
    magnitude = np.abs(normalized)
    return magnitude

# MSE between audio samples
def diff_avg(audio1,audio2):
    # Normalize
    n_audio1 = normalize(audio1)
    n_audio2 = normalize(audio2)
    
    # Diff
    diff = n_audio1 - n_audio2
    abs_diff = np.abs(diff)
    overall_change = sum(abs_diff)
    average_change = overall_change/len(audio1)
    return average_change

# L2 difference between audio samples
def diff_l2(audio1,audio2):
    # Normalize
    n_audio1 = normalize(audio1)
    n_audio2 = normalize(audio2)
    l2 = np.linalg.norm(n_audio2-n_audio1,2)
    return l2


def fft_compression(path,audio_image,factor,fs):
    '''
    # DFT Attack
    # path: path to audio file
    # Audio_image: audio file as an np.array object
    # factor: the intensity below which you want to zero out
    # fs: sample rate
    '''
    # Take FFT
    fft_image = sc.fftpack.fft(audio_image.ravel())
    
    # Zero out values below threshold
    fft_image[abs(fft_image) < factor] = 0
    
    # inverse fft
    ifft_audio = sc.fftpack.ifft(fft_image).real
    
    # New file name
    new_audio_path = path[0:-4]+'_'+str(fs)+'_FFT_'+str(factor)+'.wav'
    return new_audio_path, np.asarray(ifft_audio,dtype=np.int16)


def ssa_compression(path,audio_image,factor,fs,percent = True, pc=None,v=None):
    '''
    # SSA Attack
    # path: path to audio file
    # Audio_image: audio file as an np.array object
    # factor: the total percent of the lowest SSA componenets you want to discard
    # pc: first element that the ssa(data, window). Pass it to make execution fase
    # v: third element that the ssa(data, window). Pass it to make execution fase
    # fs: sample rate
    '''
    data = audio_image.ravel()
    window = int(len(data)*0.05)
    print('Factor Initial: '+str(factor))
    if(window>3000):
        window = 3000
    if(percent):
        factor = int((window)*factor/100)
    factor = 1 if(factor == 0) else int(factor)
    print('Factor Percent: '+str(factor))
    if type(pc) is not np.ndarray:
        pc, s, v = ssa(data, window)
    print('Factor for K: '+str(factor))
    reconstructed = inv_ssa(pc, v, np.arange(0,factor,1))
    new_audio_path = path[0:-4]+'_SSA_'+str(factor)+'.wav'
    return new_audio_path, np.asarray(reconstructed,dtype=np.float64).ravel(),pc,v

def perturb(og_audio_path,
            audio,
            atk_name,
            fs, 
            factor,
            raster_width,
            pc=None,
            v=None,
         ):
    frame = audio  
    if(atk_name == ssa_atk_name):
        return ssa_compression(og_audio_path,frame,factor,fs,pc = pc,v =v)

    elif(atk_name == fft_atk_name):
        path, perturbed_frame= fft_compression(og_audio_path,frame,factor,fs)

    return path, perturbed_frame.ravel()

def bst_atk_factor(min_atk,max_atk,val_atk,atk_name,og_label,atk_label):
    '''
    # For searching the best attack factor using binary search
    # For DCT, decrease factor if evasion success, increase other wise
    # For SSA, SVD and PCA, increase factor if evasion success, decrease other wise
    '''
    if(atk_label == og_label):
        succ = False
    else:
        succ = True
    
    init_val_atk = val_atk
    if(atk_name == dct_atk_name or atk_name == fft_atk_name or atk_name == floor_atk_name):
        if(succ):
            max_atk = val_atk
            val_atk = np.abs(min_atk+max_atk)/2
            
        else:
            min_atk = val_atk
            val_atk = np.abs(min_atk+max_atk)/2
            
    elif(atk_name == pca_atk_name or atk_name == svd_atk_name or atk_name == ssa_atk_name):
        if(succ):
            min_atk = val_atk
            val_atk = np.abs(min_atk+max_atk)/2
            
        else:
            max_atk = val_atk
            val_atk = np.abs(min_atk+max_atk)/2
            
    
    return int(min_atk),int(max_atk),int(val_atk),(init_val_atk==val_atk) 

def atk_bst(audio_path,write_location,audio_files,raster_width,models,attack,true_label):

    og_audio_path=audio_path
    _raster_width = raster_width[0]
    model =  models
    _attack_name = attack[0]
    
    # pick audio to pertur

    '''
    '''

    og_label=true_label
    print(og_label)

    #og_label=target_label

    # Read file to attack
    data=audio_files
   


    # Copy data to a new mutable variables
    perturbed_audio = np.copy(data)
    mistranscribed_audio = np.copy(data)
    # This is the frame we will be attacking
    frame_to_perturb = data

    # Need the min var for BST
    min_attack_factor = 0

    # Max factor for perturbation
    max_attack_factor = _raster_width if (_attack_name != dct_atk_name) else 8000
    max_attack_factor = max(abs(sc.fftpack.fft(data))) if (_attack_name == fft_atk_name) else max_attack_factor
    _attack_factor= max_attack_factor/2

    # For ssa
    pc = v = None

    # Maxiumu iterations
    if(_attack_name == ssa_atk_name):
        max_allowed_iterations = 1
    if(_attack_name == fft_atk_name):
        max_allowed_iterations = 15
    if(_attack_name == dct_atk_name):
        max_allowed_iterations = 15

    # Initialize iteration counter
    itr = 0

    # For each iteration
    # Generate attack sample using the _attack_factor
    # if attack works reduce max_attack_factor and min_attack_factor
    # if attack does not work increase max_attack_factor and min_attack_factor
    # Save only the attack file that works to the dataframe

    while(itr < max_allowed_iterations):
        # This variable is written to the dataframe to show the last iteration
        bst = False
        _window_size = 100
        # Attack!!
        atk_result = perturb(og_audio_path,
            audio = frame_to_perturb,
            atk_name = _attack_name,
            fs = 16000, 
            factor = _attack_factor,
            raster_width = _raster_width,
                             pc = pc,
                             v = v,
         )
        # Recycling pc and v to reduce computation time
        if(_attack_name == ssa_atk_name):
            perturbed_audio_path, perturbed_audio_frame, pc,v = atk_result
        else:
            perturbed_audio_path, perturbed_audio_frame = atk_result


        perturbed_audio_path = perturbed_audio_path[:-4]+'_BST.wav'
        perturbed_audio[0:len(perturbed_audio_frame)] = \
        perturbed_audio_frame


     

        
        # Transcribe
        perturbed_audio1=torch.from_numpy(perturbed_audio)
        perturbed_audio2=torch.reshape(perturbed_audio1,(1,16000,1))
        print(perturbed_audio1.shape)
        



        outputs1=model.check(perturbed_audio2)
        #outputs = model(batch)
        outputs = torch.nn.functional.softmax(outputs1, dim=1)

        transcribed_perturbation = outputs.data.max(1, keepdim=True)[1]
                 
        # Delete newly created audio
        

        if(og_label != transcribed_perturbation):
            mistranscribed_audio = perturbed_audio
    
        # Adjust max and min factor varaibles
        new_min_attack_factor,new_max_attack_factor,new_attack_factor,complete = \
        bst_atk_factor(min_atk = min_attack_factor ,max_atk = \
                       max_attack_factor ,val_atk = _attack_factor \
                       ,atk_name = _attack_name ,og_label = og_label ,atk_label = transcribed_perturbation)


        min_attack_factor,max_attack_factor,_attack_factor = \
            new_min_attack_factor,new_max_attack_factor,new_attack_factor 

        # Distances between original and perturbe audio file
        l2 = diff_l2(data,perturbed_audio)
        avg = diff_avg(data,perturbed_audio)
        if transcribed_perturbation is None:
            transcribed_perturbation = 'None'
        print('Iteration #: ' + str(itr+1))
        print(og_label)
        print(transcribed_perturbation)
        print('\t'*2 + 'MSE: ' + str(avg))
    


        itr = itr + 1
        if(complete): break
    print('*'*80)
    print("Complete")
    if(og_label.equal(transcribed_perturbation[0])):
        print('find')
        return 1
    else:
        return 0

    





# Okay, model 现在可以 load 了

@hydra.main(config_path="/liquid-s4/configs", config_name="config.yaml")
def main(config: OmegaConf):

    raster_width = [100]
    model_name='liquid-s4'
    audio_path='/liquid-s4/perturbed_audio_path.wav'
    write_location='/liquid-s4/perturbed_audio_path1.wav'
    config=utils.train.process_config(config)
    if config.train.seed is not None:
        pl.seed_everything(config.train.seed, workers=True)
    model = SequenceLightningModule(config)
    model=model.load_from_checkpoint('/liquid-s4/outputs/2022-12-23/23-11-02-698463/checkpoints/val/accuracy.ckpt')
    model.eval()
    raster_width = [8]
    attack = ['ssa']
    start = datetime.datetime.now()
    # Run attack


    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        #model.cuda()
    
    n_mels=32
    feature_transform = Compose([ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
    transform = Compose([LoadAudio(), FixAudioLength(), feature_transform]) 
    train_dataset_dir='/liquid-s4/data/final_data1'
    train_dataset = SpeechCommandsDataset(train_dataset_dir, transform)
    train_dataloader = DataLoader(
    train_dataset,
    batch_size=1,
    sampler=None,
    pin_memory=use_gpu,
    num_workers=4,
    drop_last=False,
    shuffle=False,
    )
    pbar = tqdm(train_dataloader, unit="audios", unit_scale=train_dataloader.batch_size)
    total_accuracy=0
    for batch in pbar:
        inputs=batch['samples']
        
        inputs = inputs.unsqueeze(1)
        targets = batch['target']
        inputs = inputs.transpose(1,2)
        print(inputs.shape)
              
        outputs1=model.check(inputs)
        outputs = torch.nn.functional.softmax(outputs1, dim=1)

        pred = outputs.data.max(1, keepdim=True)[1]
        print(pred)


        audio=inputs[0].cpu().detach().numpy().flatten()

       
        true_label = batch['target']
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        result=atk_bst(audio_path,write_location,audio,raster_width,model,attack,true_label)
       
        if (result==1):
            total_accuracy=total_accuracy+1
            print(total_accuracy)
    print(total_accuracy)


        
    
if __name__ == '__main__':
    
    
    # Path to audio files
        main()