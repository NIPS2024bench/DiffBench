# Point Cloud Generation
#### Please refer to https://github.com/luost26/diffusion-point-cloud as the generator
#### After installation for it.
#### python test_gen.py --ckpt ./pretrained/GEN_airplane.pt --categories airplane python test_gen.py --ckpt ./pretrained/GEN_chair.pt --categories chair for airplane and chair correspondingly. 



#### Point Cloud score evaluation
#### For HyCORE model evaluation, please install https://github.com/diegovalsesia/hycore first and copy the great_evaluation_pointcloud_hycore into the classfication_ModelNet40 directory, we prepare a dataloader named shapenet1024 in the code with comment, please copy and paste it into data.py\


#### For  pointnet model, we use the same dataloader, please utilize the same dataloader and copy great_evaluation_pointcloud_pointnet.py inyo classification_ModelNet40 directory.

#### Adversarial Attack
#### For adversarial attack of it, we ref to  SI-ADV https://github.com/shikiw/SI-Adv  as the attack method, please install it and paste main, attacks_hycore and main_hycore into it, use main_hycore.py to evaluate the hycore model and main.py with specified model name to evaluate the modelnet model.