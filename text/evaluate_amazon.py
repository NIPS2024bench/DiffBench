import json
import requests
import time
import numpy as np
import datetime
import math
headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiYjM0MTAyMGItM2E1MC00YmEyLWI0NjktYjhmYjE5MzFmOTlkIiwidHlwZSI6ImFwaV90b2tlbiJ9.pRiMcey7mCyZAQx0tGSRTwf8UPZXrQ1Dq9C7kY1q4ss"}

url ="https://api.edenai.run/v2/text/sentiment_analysis"
starttime = datetime.datetime.now()

text_name='/LatentOps/ckpts/large_yelp/sample/negative.txt'

f=open(text_name)

line=f.readline()
store_list=[]

while line:
  
    url1=line
    payload={"providers": "amazon", 'language': "en", 'text': url1,"response_as_dict": True,"attributes_as_list": False,"show_original_response": True}
    response = requests.post(url, json=payload, headers=headers)
    #print(response)
    result = json.loads(response.text)
    if response.status_code == 200:

        print(result)
        store_list.append(result)
        #print(result['ibm']['items'])
    else:
        time.sleep(5)
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
          
            
            store_list.append(result)
            #print(result)
            #print(result['ibm']['items'])
        else:
            store_list.append(0)
            print(114514)
    line=f.readline()


number_imgs=500


result1=[]
difference=np.zeros(500,dtype=float)

for i in range(number_imgs):
    label='NEGATIVE'
    response=store_list[i]
    if response==[]:
        difference[i]=0
    else:
        senti_label=response['amazon']['original_response']['Sentiment']
        print(senti_label)
        if senti_label=='NEGATIVE':
               
                Positive_score=response['amazon']['original_response']['SentimentScore']['Positive']
                Negative_score=response['amazon']['original_response']['SentimentScore']['Negative']
                difference[i]=math.sqrt(
                            math.pi/2)*(Negative_score-Positive_score)
        else:
                difference[i]=0

print(np.mean(difference))

endtime = datetime.datetime.now()
print ((endtime - starttime).seconds)