import copy
import os
import random
import sys
sys.path.append(r'liquid-s4')
import datetime
import time
from functools import partial, wraps
from typing import Callable, List, Optional

import hydra
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
import math


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

sys.path.append(r'state-spaces/sashimi/sc09_classifier')

from audio_utils import SequenceLightningModule,SpeechCommandsDataset


CLASSES = 'zero, one, two, three, four, five, six, seven, eight, nine'.split(', ')


from transforms import FixAudioLength, LoadAudio, ToMelSpectrogram, ToTensor

@hydra.main(config_path="liquid-s4/configs", config_name="config.yaml")
def main(config: OmegaConf):

    # Process config:
    # - register evaluation resolver
    # - filter out keys used only for interpolation
    # - optional hooks, including disabling python warnings or debug friendly configuration
    config = utils.train.process_config(config)

    # Pretty print config using Rich library
    if config.train.seed is not None:
        pl.seed_everything(config.train.seed, workers=True)
    

    model = SequenceLightningModule(config)
 
    model=model.load_from_checkpoint('Downloads/liquid-s4/outputs/2022-12-23/23-11-02-698463/checkpoints/val/accuracy.ckpt')

    model.eval()

   
   

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        model.cuda()
    
    n_mels=32
    feature_transform = Compose([ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
    transform = Compose([LoadAudio(), FixAudioLength(), feature_transform]) 
    train_dataset_dir='Downloads/liquid-s4/data/final_data1'
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
    
    difference=np.zeros(2000,dtype=float)
    j=0
    for batch in pbar:
        inputs=batch['samples']
        inputs = inputs.unsqueeze(1)
        inputs = inputs.transpose(1,2)
        true_label = batch['target']
        

        if use_gpu:
            inputs = inputs.cuda()
            true_label = true_label.cuda()

        indices=torch.tensor([34,27,32,31,22,3,29,28,20,25]).cuda()
        
        
        outputs1 = model.check(inputs)
        
        logits=outputs1
        logits=torch.index_select(logits[0],0,indices)
        #print(logits)
        outputs = torch.sigmoid(logits)

        out1=outputs.detach().cpu().numpy()
        top2_label = np.argsort(out1)[-2]
        predicted_class_id = outputs.data.max(0, keepdim=True)[1]
        predicted_label=indices[predicted_class_id]
        #print(true_label)
        #print(predicted_label)
        if true_label.equal(predicted_label):
            difference[j]=math.sqrt(
                            math.pi/2)*(outputs[predicted_class_id]-outputs[top2_label])
            print(j)
            j=j+1
          
        else:
            difference[j]=0
            j=j+1
    average_score=np.mean(difference)
    print(average_score)
    






if __name__ == "__main__":
    starttime = datetime.datetime.now()
    main()
    endtime = datetime.datetime.now()
    print ((endtime - starttime).seconds)