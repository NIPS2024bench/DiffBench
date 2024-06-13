# Text generation
### Please refer to Latentops https://github.com/guangyliu/LatentOps  , after install it and enter.
#### cd code
#### bash conditional_generation.sh $1 $2 then you should have the generated txt as described in the repository

# Text score evaluation
### Please install https://github.com/huggingface/transformers first,then 
### python  great_evaluation.py  --model-name bert-base-uncased or  great_evaluation.py  --model-name xlnet-large-cased for bert and xlnet correspondingly.


# Text Attack 

### Please install Text Attack https://github.com/QData/TextAttack first
#### Then you need to copy the test_dataset.py to the textattack folder, which is used to load the data from text file.
#### Assuming you have two cards. CUDA_VISIBLE_DEVICES=0,1 textattack attack --model-from-huggingface bert-base-uncased/xlnet-large-cased --recipe textbugger --dataset-from-file test_dataset.py --num-examples 500 --parallel 





# Text API evaluation:

#### Step 1: Please regist in https://www.edenai.co/ and get your own token for loading the sentiment analysis API
#### Step 2: To evaluate the score, please change the load_dir to your txt file.
#### Step 3: python evaluate_amazon.py for amazon and evaluate_microsoft.py for microsoft.

# Text attack: