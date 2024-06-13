# Suppose x as an image from tensor

import torch
from torch import nn
import requests
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
request_url = "https://aip.baidubce.com/rest/2.0/face/v3/detect"
access_token = '24.856fefd9b3d88f4fbf5976f8174b4695.2592000.1681454985.282335-27820666' # input your own access_token
headers = {'content-type': 'application/json'}
device="cuda"
import base64
import numpy as np
from torch.autograd import Variable
from deepface import DeepFace
request_url = request_url + "?access_token=" + access_token
import cv2
from io import BytesIO
from deepface.commons import functions, realtime, distance as dst

import torchvision.transforms as transforms

loader = transforms.Compose([transforms.ToTensor()]) 
 
unloader = transforms.ToPILImage()



class Face(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, image):
        image = torch.Tensor.cpu(image).clone()
        image = image.squeeze(0)
        image = unloader(image)
        img_buffer = BytesIO()    
        image.save(img_buffer, format='JPEG')    
        byte_data = img_buffer.getvalue()    
        base64_str = base64.b64encode(byte_data)    
        image=base64_str
        img = image.decode('utf-8')   
        params = {"image": img,"image_type":"BASE64","face_field":"age,beauty,glasses,gender,race"}
        response = requests.post(request_url, data=params, headers=headers).json()
        print(response)
        if response['error_code']!=0:
            logits_score=torch.Tensor([0.5])
            counter=1-logits_score
            logits_score=Variable(logits_score,requires_grad=True)
            counter=Variable(counter,requires_grad=False)
            
            a=torch.Tensor([[logits_score, counter]]).to(device)
            a=Variable(a,requires_grad=True)
            
            return a
        else:
            gender_label=response['result']['face_list'][0]['gender']['type']
            #print(gender_label)
            if gender_label=='female':
                    logits_score=response['result']['face_list'][0]['gender']['probability']
            else:
                    logits_score=1-response['result']['face_list'][0]['gender']['probability']
            logits_score=torch.Tensor([logits_score])
            counter=1-logits_score
          

            logits_score=Variable(logits_score,requires_grad=True)
            counter=Variable(counter,requires_grad=False)
            

            a = torch.Tensor([[logits_score, counter]]).to(device)
            a=Variable(a,requires_grad=True)
            
            return a


class Deepfacemodel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, image):
        if image.shape[0]==0:
            return torch.Tensor([[0.5, 0.5]]).to(device)
        image_tensor=image[0]
        image_numpy_form_tensor = image_tensor.mul(255).byte().cpu() #将tensor中的所有像素点的值乘 255
        #print(image_numpy_form_tensor.shape)
        image_numpy_form_tensor = np.array(image_numpy_form_tensor).transpose(1,2,0)
        image_numpy_form_tensor = cv2.cvtColor(image_numpy_form_tensor,cv2.COLOR_RGB2BGR)
        
        #image=image_numpy_form_tensor
        #image=np.array(image)
        obj=DeepFace.analyze(img_path = image_numpy_form_tensor, actions = ['gender'], enforce_detection=False,silent=True)
        woman_score=obj[0]['gender']['Woman']
        man_score=obj[0]['gender']['Man']
        a = torch.Tensor([[woman_score, man_score]]).to(device)
        #a=Variable(a,requires_grad=True)
            
        

        return a


from deepface.basemodels import VGGFace
import os
from pathlib import Path
import gdown
import numpy as np

from deepface.commons import functions

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])

if tf_version == 1:
	from keras.models import Model, Sequential
	from keras.layers import Convolution2D, Flatten, Activation
elif tf_version == 2:
	from tensorflow.keras.models import Model, Sequential
	from tensorflow.keras.layers import Convolution2D, Flatten, Activation

#url = 'https://drive.google.com/uc?id=1wUXRVlbsni2FN9-jkS_f4UTUrm1bRLyk'


def loadModel(url = 'https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5'):

	model = VGGFace.baseModel()

	#--------------------------

	classes = 2
	base_model_output = Sequential()
	base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
	base_model_output = Flatten()(base_model_output)
	base_model_output = Activation('softmax')(base_model_output)

	#--------------------------

	gender_model = Model(inputs=model.input, outputs=base_model_output)

	#--------------------------

	#load weights

	home = functions.get_deepface_home()

	if os.path.isfile(home+'/.deepface/weights/gender_model_weights.h5') != True:
		print("gender_model_weights.h5 will be downloaded...")

		output = home+'/.deepface/weights/gender_model_weights.h5'
		gdown.download(url, output, quiet=False)

	gender_model.load_weights(home+'/.deepface/weights/gender_model_weights.h5')

	return gender_model

	#--------------------------


class MobileNetV2(tf.keras.Model):
    def __init__(self):
        super().__init__()
       
        self.model=loadModel()
        
    def call(self, image):
        detector_backend = 'opencv'
        enforce_detection=True
        align = True
        silent = False
        
        image_tensor=image
        image_numpy_form_tensor = image_tensor #将tensor中的所有像素点的值乘 255
        #print(image_numpy_form_tensor.shape)
        image_numpy_form_tensor = np.array(image_numpy_form_tensor).transpose(1,2,0)
        
        image=image_numpy_form_tensor
        image=image*255
         
        print(image.shape)

        img_content, img_region, img_confidence  = functions.extract_faces(img=image, target_size=(224, 224), detector_backend=detector_backend, grayscale = False, enforce_detection=enforce_detection, align=align)
        x = self.model(img_content)

       
        
        return x







