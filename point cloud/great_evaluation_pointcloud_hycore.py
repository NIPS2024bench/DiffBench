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

'''
class ShapeNet1024Total(Dataset):
    def __init__(self, num_points, partition='train'):
        rng1=np.load('/diffusion-point-cloud/results/GEN_Ours_airplane_1672636027/out.npy')
        rng2=np.load('/diffusion-point-cloud/results/GEN_Ours_chair_1672636121/out.npy')
        
        rng1=rng1[:500]
        len1=rng1.shape[0]
        rng2=rng2[:500]

        rng=np.concatenate((rng1,rng2),axis=0)
        self.data=rng
        len1=self.data.shape[0]
        len2=500

        label1=torch.zeros(len2,dtype=torch.int32)
        #label1=torch.ones(len2,dtype=torch.int64)
        label2=8*torch.ones(len2,dtype=torch.int32) 

        label=torch.cat((label1,label2),axis=0)
        self.label=label
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]

        if self.partition == 'train':
            # pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


'''

from data import ModelNet40,ShapeNet1024Total
import models as models
from models.pointmlp import Hype_pointMLP

model_names = sorted(name for name in models.__dict__
                     if callable(models.__dict__[name]))


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
    net = Hype_pointMLP()
    net = net.to(device)

    test_loader = DataLoader(ShapeNet1024Total(partition='train', num_points=1024), num_workers=4,
                                batch_size=1, shuffle=True, drop_last=True) 
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net.load_state_dict(checkpoint['net'])


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
    #end1 = datetime.datetime.now()
    #print('time:({} ms)'.format(int((end1-start1).total_seconds())),file=data)

if __name__ == "__main__":
    starttime = datetime.datetime.now()
    args = get_arg()
    evaluate(args)
    endtime = datetime.datetime.now()
    print ((endtime - starttime).seconds)