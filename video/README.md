# For Video Generation:

## Step 1:  install https://github.com/VideoCrafter/VideoCrafter and corresponding environment with given checkpoint
## Step 2: copy generate.py in this file to VideoCrafter and python generate.py with altering the prompts in the appendix for 5 different labels
## Step 3: Now you should have the videos in an output dir

# For mp4 to frames:

### you can refer to prepare_ucf101 in mmaction2 toolkit to load the classifiers and extract frames from the videos.

# For score inference:

## First, you should prepare an annotation file with same format as orginal annotation file required by mmaction2
### eg: filename frame_length video_label for each line

### Step 1 : install https://github.com/sli057/Geo-TRAP and corresponding environment with given checkpoint
### Step 2: copy and paste great_score_py and decompose_query.py into query_attack folder and c3d_ucf_...py into config/recognation/c3d 
#### Step 3: modify the data_path and annotation file in c3d_ucf..py
### Step 4: To evaluate score: slowfast: python query_attack/great_score.py --config configs/recognition/c3d/c3d_ucf_MotionPerturbation1.py --config_rec configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_ucf_rgb.py --checkpoint_rec work_dirs/slowfast_r50_video_3d_4x16x1_256e_ucf101_1_rgb/epoch_30.pth                                   --work-dir decompose_query_version1                                  --transform_type_query  translation_dilation


#### i3d: python query_attack/great_score.py --config configs/recognition/c3d/c3d_ucf_MotionPerturbation1.py --config_rec configs/recognition/i3d/i3d_r50_32x2x1_100e_ucf_rgb.py  --checkpoint_rec work_dirs/i3d_r50_32x2x1_100e_ucf_split_1_rgb/epoch_100.pth                         --work-dir decompose_query_version1                                  --transform_type_query  translation_dilation

#### c3d: python query_attack/great_score.py --config configs/recognition/c3d/c3d_ucf_MotionPerturbation1.py --config_rec configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py --checkpoint_rec work_dirs/c3d_sports1m_16x1x1_45e_ucf101_rgb_20201021-26655025.pth                         --work-dir decompose_query_version1                                  --transform_type_query  translation_dilation



# For adversarial attack verification


#### slowfast: python query_attack/decompose_query.py --config configs/recognition/c3d/c3d_ucf_MotionPerturbation1.py --config_rec configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_ucf_rgb.py --checkpoint_rec work_dirs/slowfast_r50_video_3d_4x16x1_256e_ucf101_1_rgb/epoch_30.pth                                   --work-dir decompose_query_version1                                  --transform_type_query  translation_dilation


#### i3d: python query_attack/decompose_query.py --config configs/recognition/c3d/c3d_ucf_MotionPerturbation1.py --config_rec configs/recognition/i3d/i3d_r50_32x2x1_100e_ucf_rgb.py  --checkpoint_rec work_dirs/i3d_r50_32x2x1_100e_ucf_split_1_rgb/epoch_100.pth                         --work-dir decompose_query_version1                                  --transform_type_query  translation_dilation

#### c3d: python query_attack/decompose_query.py --config configs/recognition/c3d/c3d_ucf_MotionPerturbation1.py --config_rec configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py --checkpoint_rec work_dirs/c3d_sports1m_16x1x1_45e_ucf101_rgb_20201021-26655025.pth                         --work-dir decompose_query_version1                  

