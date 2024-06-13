import os 
import textattack

data_path1='/LatentOps/ckpts/large_yelp/sample/positive.txt'

data=[]


with open(data_path1,'r') as f:
        for line in f:
            data.append((line.strip('\n'),0))
#print(data)

dataset=textattack.datasets.Dataset(data)
