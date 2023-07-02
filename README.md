# Contrastive-Hierarchical-Clustering
This is the official code for the paper ["Contrastive Hierarchical Clustering"](https://arxiv.org/pdf/2303.03389.pdf) (ECML PKDD 2023).
Our repository is build on top of the [SimCLR implementation](https://github.com/leftthomas/SimCLR) done by [leftthomas](https://github.com/leftthomas)

## Environment setup
Our code runs on a single GPU. It does not support multi-GPUs.
The code is compatible with `Pytorch >= 1.7`. See requirements.txt for all prerequisites, and you can also install them using the following command.
```
conda create --name <env> --file requirements.txt
```

## Training
For every dataset configuration file `cfg/<datasetname>` specifies model architectures and hyperparameters for the training.
```
python main.py --dataset-name cifar10 # Start training for CIFAR10
python main.py --dataset-name cifar100 # Start training for CIFAR100
python main.py --dataset-name stl10 # Start training for STL10
python main.py --dataset-name imagenet10 # Start training for ImageNet10
python main.py --dataset-name imagenetdogs # Start training for ImageNetDogs
```

## Evaluation and pretrained models
After the training, model can be evaluate with `evaulate.py` to generate NMI, ARI and ACC metrics. 
Pretrained models are available in `pre-traiend` folder.
```
python evaluate.py --save_point ./pre-trained/CIFAR10/models/ --dataset-name cifar10 # Generate metrics for CIFAR10
python evaluate.py --save_point ./pre-trained/STL10/models/ --dataset-name stl10 # Generate metrics for STL10
python evaluate.py --save_point ./pre-trained/ImageNet10/models/ --dataset-name imagenet10 # Generate metrics for ImageNet10
python evaluate.py --save_point ./pre-trained/ImageNetDogs/models/ --dataset-name imagenetdogs # Generate metrics for ImageNetDogs
```