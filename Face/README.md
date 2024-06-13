# For Image Generation
### Please install stable-diffusion version2 https://github.com/Stability-AI/stablediffusion first, after installation, please use following command to generate the images
#### python scripts/txt2img.py --prompt "a professional photograph of a man’s detailed face," --ckpt '/stablediffusion/checkpoints/512-base-ema.ckpt' --config '/stablediffusion/configs/stable-diffusion/v2-inference.yaml' --device cuda --outdir 1



# For Score evaluation
#### Install deepface https://github.com/serengil/deepface first 
#### python deep_face_score_man.py, you need to change the file path to your own data path.


# For Square Attack:
#### python evaluate_old1.py with installation for autoattack first.

