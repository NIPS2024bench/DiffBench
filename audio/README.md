# Audio Generation
### Please refer to https://github.com/lmnt-com/diffwave to generate audios
#### After you have got the audios, please follow the guideline in state-spaces/sashimi/ to generate label for audios
#### python test_speech_commands.py --sample-dir your_sample_dir --save-probs resnext.pth
#### python prepare_sc09.py --methods speeches --cache_dir state-spaces/sashimi/sc09_classifier/cache  --sample_dir where you want to store the processed samples



### Audio Score Estimation
#### We use liquid-s4 https://github.com/raminmh/liquid-s4 and s4 https://github.com/HazyResearch/state-spaces as the evaluated classifiers, please refer to the command provided in the original repo to train the classifiers for speech command dataset

#### We provide the score evaluation script, python great_evaluation_audio_liquid.py for liquid-s4 and python great_evaluation_audio_state.py  for s4
####  You have 3 things to modify:
#### Line 71: you need to change the config path for hydra to the config you use to train the models
#### Line 102: you need to change the load_dir to the dataset you store the samples
#### Line 87: you need to change the checkpoints to the one you trained.


#### Audio Adversarial Attack:
#### python my_attack_liquid.py and python my_attack_sequence.py for liquid-s4 and s4 correspondingly.
#### change the correspond parts like in the audio score estimation session