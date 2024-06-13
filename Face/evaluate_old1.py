import torch
from torch import nn
import requests
import os
from model import loadModel,MobileNetV2,Deepfacemodel
import numpy as np
from utils import read_directory
import base64
from deepface import DeepFace

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

directory_name='/stablediffusion/old_woman2/samples'
device="cuda"

a=read_directory(directory_name)
a = np.transpose(a, (0,3, 1, 2))
print(type(a))

directory_name1='/stablediffusion/old_man/samples'
b=read_directory(directory_name)
b = np.transpose(b, (0,3, 1, 2))


a=torch.cat([a,b],axis=0)

#print(a[1].shape)
image=a[1]
'''
image=np.array(image)
obj=DeepFace.analyze(img_path = image, 
        actions = ['gender'], enforce_detection=False
)
print(obj[0]['gender'])
'''
y=np.load('/autoattack-face/old.npy')
print(type(y))
#print(len(y))
y[0:250]=0
y[250:500]=1
model_face=Deepfacemodel().to(device)

#print(y[3])

#print(model_face(a[3]))
'''
num=0
for i in range(50):
        if(model_face(a[i]).max(1)[1]==y[i]):
                num=num+1

print(num)
'''
'''
for i in range(500):
        print(model_face(a[i]))
'''


from autoattack import AutoAttack

epsilon=2
adversary = AutoAttack(model_face, norm='L2', eps=epsilon,seed=0)
adversary.attacks_to_run = ['square']

young_list=[40,153,157,188,216,295,382,412]
#images=a[~young_list]
young_list=[295,382,412]

images=torch.cat((a[0:40],a[41:153], a[154:157],a[158:188],a[189:216],a[217:295],a[296:382],a[383:412],a[413:500]), 0)
print(len(images))

y=torch.from_numpy(y)

labels=torch.cat((y[0:40],y[41:153], y[154:157],y[158:188],y[189:216],y[217:295],y[296:382],y[383:412],y[413:500]), 0)

print(len(labels))

x_adv = adversary.run_standard_evaluation_individual(images.to(device), labels.to(device), bs=1)



'''
for i in range(189,250):
        images=a[i:i+1]
        labels=torch.from_numpy(y)[i:i+1]
        print(i)
        x_adv = adversary.run_standard_evaluation_individual(images.to(device), labels.to(device), bs=1)

'''
'''
images=a[0:500]
labels=torch.from_numpy(y)[0:500]
adversary.attacks_to_run = ['square']
x_adv = adversary.run_standard_evaluation_individual(images.to(device), labels.to(device), bs=1)
'''