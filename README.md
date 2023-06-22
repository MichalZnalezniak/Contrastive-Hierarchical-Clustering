# Contrastive-Hierarchical-Clustering
## Enviroment setup
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
python evaluate.py --save_point ./pre-trained/CIFAR100/models/ --dataset-name cifar100 # Generate metrics for CIFAR100
python evaluate.py --save_point ./pre-trained/STL10/models/ --dataset-name stl10 # Generate metrics for STL10
python evaluate.py --save_point ./pre-trained/ImageNet10/models/ --dataset-name imagenet10 # Generate metrics for ImageNet10
python evaluate.py --save_point ./pre-trained/ImageNetDogs/models/ --dataset-name imagenetdogs # Generate metrics for ImageNetDogs
```