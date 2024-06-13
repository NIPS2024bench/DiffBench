# We want to define a function that receives the directory of images and output NCHW format np array, we may need to modify the model to let them accept the array.


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import numpy as np
import torch
from io import BytesIO
 # this if for store all of the image data
import torchvision.transforms as transforms

loader = transforms.Compose([transforms.ToTensor()]) 
 
unloader = transforms.ToPILImage()

import base64
# this function is for read image,the input is directory name
def read_directory(directory_name):
    array_of_img = []
    dir=directory_name
    imgList = os.listdir(dir)
    #print(imgList)
    imgList.sort(key=lambda x: int(x.replace("frame","").split('.')[0]))
    j=0
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for count in range(0,len(imgList)):
        #print(filename) #just for test
        #img is used to store the image data 
        
        im_name = imgList[count]
        im_path = os.path.join(dir,im_name)


        img = cv2.imread(im_path)
        
        img = cv2.resize(img,(224,224))
        #img=cv2.resize
        img = img / 255
        
       
        #transf = transforms.ToTensor()
        #img_tensor = transf(img)  # tensor数据格式是torch(C,H,W)
        array_of_img.append(img)
        #print(img)
    #print(type(array_of_img))
    final_array=np.array(array_of_img)
    final_array=torch.from_numpy(final_array)
    #final_array=torch.Tensor(array_of_img)
    print(final_array.shape)

    return final_array



#directory_name='test'
##a=read_directory(directory_name)
'''
a = np.transpose(a, (0,3, 1, 2))
print(a.shape)
image=a[0]
image = torch.Tensor.cpu(image).clone()
image = image.squeeze(0)
image = unloader(image)



img_buffer = BytesIO()    
image.save(img_buffer, format='JPEG')    
byte_data = img_buffer.getvalue()    
base64_str = base64.b64encode(byte_data)    
image=base64_str
img = image.decode('utf-8')   

print(img)
'''


