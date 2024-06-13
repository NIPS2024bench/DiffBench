from deepface import DeepFace

# 0是female

import os
import requests
import numpy as np
import math
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

starttime = datetime.datetime.now()
online_api_name='deepface'


h=0 # 0 to 3


attribute_name_list=['old','young','with-eyeglasses','without-eyeglasses','overall']
attribute_name=attribute_name_list[h]

load_file_directory=['/home/zaitang/interfacegan/results/age-conditioned-old-balanced','/home/zaitang/interfacegan/results/stylegan_celebahq_gender_editing_young-balanced','/home/zaitang/interfacegan/results/with-eyeglasses1-balanced','/home/zaitang/interfacegan/results/without-eyeglasses-balanced']

load_img_name=load_file_directory[h]

load_file_directory=['/stablediffusion/old_woman2/samples','/stablediffusion/old_man/samples','/stablediffusion/young_man/samples','/stablediffusion/young_woman/samples']
#load_file_directory=['/stablediffusion/without_woman/samples','/stablediffusion/without_man/samples']
load_img_name=load_file_directory[h]

store_list=[]


dir = load_img_name
imgList = os.listdir(dir)
#print(imgList)
imgList.sort(key=lambda x: int(x.replace("frame","").split('.')[0]))

#print(imgList)
#for count in range(0, len(imgList)):
for count in range(0,250):
    im_name = imgList[count]
    im_path = os.path.join(dir,im_name)
    print(im_path)
    obj=DeepFace.analyze(img_path = im_path, 
        actions = ['gender'], enforce_detection=False
)
    print(obj[0]['dominant_gender'])

    store_list.append(obj)

a=np.array(store_list)
difference=np.zeros(250,dtype=float)
correct=0
for i in range(250):
    gender='Woman'
    response=a[i]
    woman_score=response[0]['gender']['Woman']/100
    man_score=1-woman_score
    predict_label=response[0]['dominant_gender']
   
    if predict_label==gender:
        correct=correct+1
        print(correct)
        difference[i]=math.sqrt(
                            math.pi/2)*(woman_score-man_score)
    else:
                difference[i]=0


print(np.mean(difference))
print(correct)
endtime = datetime.datetime.now()
print ((endtime - starttime).seconds)

'''

elif online_api_name=='deepface':
		if attribute_name1=="Gender":
			#for different name we use different index
				response=load_file[i]
				logits_score=response['gender']['Woman']/100

'''