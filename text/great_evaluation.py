# We specify this module for txt first. nlp-evaluation env with 11.3 cuda
from ast import parse
from unicodedata import name
import torch
from transformers import BertTokenizer, BertForSequenceClassification,XLNetTokenizer, XLNetForSequenceClassification
from transformers import pipeline
import os 
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import numpy as np
device = torch.device("cuda")
from torch.autograd import Variable
import dill
import argparse
import torch.nn.functional as F
import random

import gc
import pandas as pd
import seaborn as sns

import math
import sys

import warnings


'''
create a model list and the name can be directly used to load.
'''

model_list=[]

def get_arg():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter) 
    parser.add_argument("--n_s", type=int, default=1000, help='the generated model sample size') 
    parser.add_argument('--data_path', type=str,
                        default='/LatentOps/ckpts/large_yelp/sample/sampling_sentiment_1.0.txt', help='the samples data path')
    parser.add_argument('--log_file', type=str, default='outputs/gan.txt', help='the output file to store the result')
    parser.add_argument('--model-name', type=str,default='bert-large-uncased', choices=['xlnet-base-cased','xlnet-large-cased','bert-large-uncased','bert-base-uncased'],help='the evaluated model name')
    args = parser.parse_args()
    return args



'''
we need to leave space in the main function ,the pre_process should receive the 
data_path and return with x_data and y_data
'''

'''
For nlp task, I want to return a list
'''


def pre_process():
    
    result=[]
    
    data_path1='/LatentOps/ckpts/large_yelp/sample/negative.txt'
    data_path2='/LatentOps/ckpts/large_yelp/sample/positive.txt'
    with open(data_path1,'r') as f:
        for line in f:
            result.append(list(line.strip('\n').split(',')))
    

    with open(data_path2,'r') as f:
        for line in f:
            result.append(list(line.strip('\n').split(',')))
    
    #x_data=torch.tensor(result)
    x_data=result


    
    y_data2=torch.zeros(500,dtype=torch.int64) 
    y_data1=torch.ones(500,dtype=torch.int64)
    y_data=torch.cat((y_data1,y_data2),axis=0)



    
    return x_data,y_data


def evaluate(args):
    data_path=args.data_path
    n_s=1000

    log_file=args.log_file
    model_name=args.model_name
    x_data,y_data=pre_process()
    

    difference=np.zeros(1000,dtype=float)

    if model_name=='bert-base-uncased' or model_name=='bert-large-uncased':
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name)
    else:
        tokenizer = XLNetTokenizer.from_pretrained(model_name)
        model = XLNetForSequenceClassification.from_pretrained(model_name)




    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    j=0
    for idx in range(n_s):


    
        inputs = tokenizer(str(x_data[idx]), return_tensors="pt")
        true_label=int(y_data[idx])
        with torch.no_grad():
            logits = model(**inputs).logits
        # we have logits as the result, how to process it?
        logits=torch.sigmoid(logits)
        print(logits)

        predicted_class_id = logits.argmax().item()
        top2_id=logits.argmin().item()
        
        
        predicted_label=model.config.id2label[predicted_class_id]

        
        
        if true_label != predicted_class_id:
            difference[j]=0
            j=j+1
                
        else:
            difference[j]=math.sqrt(
                            math.pi/2)*(logits[0][predicted_class_id]-logits[0][top2_id])
            print(difference[j])
            j=j+1
    
    average_score=np.mean(difference)
    print(average_score)
    


    


    



if __name__ == "__main__":
    starttime = datetime.datetime.now()
    args = get_arg()
    evaluate(args)
    endtime = datetime.datetime.now()
    print ((endtime - starttime).seconds)
