import os
from pickle import FALSE
import sys
import numpy as np
from collections import Iterable
import importlib
import open3d as o3d

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

from baselines import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model/classifier'))

# We specify this module for txt first. nlp-evaluation env with 11.3 cuda

# pointcloud may be different from txt, but only with two classifiers in the
# same environment
from ast import parse
from unicodedata import name
import torch
import os 
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
device = torch.device("cuda")
from torch.autograd import Variable
import dill
import argparse
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random

import gc
import pandas as pd
import seaborn as sns

import math
import sys

from torch import nn

import warnings




import glob
import h5py
from torch.utils.data import Dataset
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
#from data import ShapeNet1024


from torch.utils.data import DataLoader





#sys.path.append(r'/pointMLP-pytorch/classification_ModelNet40')
sys.path.append(r'/HyCoRe/classification_ModelNet40')
from data import ModelNet40,ShapeNet1024Total
#import models as models
#from models.pointmlp import Hype_pointMLP



def load_models(classifier, model_name):
    """Load white-box surrogate model and black-box target model.
    """

  
    checkpoint = torch.load('/SI-Adv/checkpoint/checkpoint/ModelNet40/pointnet_cls.pth')
    try:
        if 'model_state_dict' in checkpoint:
            classifier.load_state_dict(checkpoint['model_state_dict'])
        elif 'model_state' in checkpoint:
            classifier.load_state_dict(checkpoint['model_state'])
        else:
            classifier.load_state_dict(checkpoint)
    except:
        classifier = nn.DataParallel(classifier)
        classifier.load_state_dict(checkpoint)
    return classifier















'''
create a model list and the name can be directly used to load.
'''

model_list=[]

def get_arg():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter) 
    parser.add_argument("--n_s", type=int, default=1000, help='the generated model sample size') 
    parser.add_argument('--data_path', type=str,
                        default='samples/samples.npz', help='the samples data path')
    parser.add_argument('--log_file', type=str, default='outputs/gan.txt', help='the output file to store the result')
    parser.add_argument('--model', type=str, default='Hype_PointNet', choices=['Hype_PointNet','pointMLP'],help='the model name for evaluate to load the ckpt')
    args = parser.parse_args()
    return args



'''
we need to leave space in the main function ,the pre_process should receive the 
data_path and return with x_data and y_data
'''

'''
For nlp task, I want to return a list
'''



def evaluate(args):
    
    print('==> Preparing data..')

    MODEL = importlib.import_module('pointnet_cls')
    classifier = MODEL.get_model(
        40,
        normal_channel=False
    )
   
    # load model weights
    classifier = load_models(classifier,'pointnet_cls')

    data_path=args.data_path
    n_s=args.n_s
    log_file=args.log_file
    model_name=args.model

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"==> Using device: {device}")


    '''
    Where we load the checkpoint function
    '''

    checkpoint='/HyCoRe/classification_ModelNet40/checkpoints/Hype_PointNet-Offv_pointmlp_hycore_var-22/best_checkpoint.pth'
    #checkpoint='/HyCoRe/classification_ModelNet40/checkpoints/Hype_PointNet-Offv_pointmlp_hycore_var-22/last_checkpoint.pth'
    net=classifier
    net = net.to(device)

    test_loader = DataLoader(ShapeNet1024Total(partition='train', num_points=1024), num_workers=4,
                                batch_size=1, shuffle=True, drop_last=True) 


    '''
    Now we have net as our model
    '''









    difference=np.zeros(1000,dtype=float)
    j=0
    
    net.eval()
    m = nn.Softmax(dim=0)
    num_correct= 0
    with torch.no_grad():
        for batch_idx,(data,label) in enumerate(test_loader):
            
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits = net(data)
              
            #indices=torch.tensor([0,8]).to(device)
            #logits=torch.index_select(logits[0],0,indices)
           
            logits=torch.sigmoid(logits[0])
            print(logits)

            predicted_label=logits.max(dim=0)[1]
            print(label)
            print(predicted_label)
            #top2_label=logits.min(dim=0)[1]
            out1=logits.detach().cpu().numpy()

            top2_label = np.argsort(out1)[-2]
            print(top2_label)

            true_label=label

            if true_label.eq(predicted_label) :
                difference[j]=math.sqrt(
                                math.pi/2)*(logits[predicted_label]-logits[top2_label])
                print(difference[j])
                j=j+1
        
                
                    
            else:
                difference[j]=0
                j=j+1
            num_correct += torch.eq(predicted_label, true_label).sum().float().item()
                

    average_score=np.mean(difference)
    print(num_correct)
    print(average_score)


if __name__ == "__main__":
    starttime = datetime.datetime.now()
    args = get_arg()
    evaluate(args)
    endtime = datetime.datetime.now()
    print ((endtime - starttime).seconds)