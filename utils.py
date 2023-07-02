from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageNet, MNIST, FashionMNIST
from torch.utils.data import Subset
import numpy as np
from torch.utils.data import ConcatDataset

def get_transforms(name):
    train_transforms = {
        'cifar10': transforms.Compose([
                        transforms.RandomResizedCrop(32),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]),

        'cifar100': transforms.Compose([
                        transforms.RandomResizedCrop(32),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])]),

        'stl10' : transforms.Compose([
                        transforms.RandomResizedCrop(96),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])]),

        'imagenet10': transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

        'imagenetdogs': transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), 
    }

    valid_transforms = {
        'cifar10': transforms.Compose([
                        transforms.Resize(32),
                        transforms.ToTensor(),
                        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]),
        'cifar100': transforms.Compose([
                        transforms.Resize(32),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])]),
        'stl10': transforms.Compose([
                        transforms.Resize(96),
                        transforms.ToTensor(),
                        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])]),
        'imagenet10': transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'imagenetdogs': transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    }
    return train_transforms[name], valid_transforms[name]




def get_contrastive_dataset(name):

    download_mapping = {
        'cifar10' : False,
        'cifar100' : False,
        'stl10' : False,
        'imagenet10' : False,
        'imagenetdogs' : False,
    }
    # Only download the chosen dataset
    download_mapping[name] = True

    train_data = {
        'cifar10': None if download_mapping['cifar10'] == False else CIFAR10Pair(root='data', train=True, transform=get_transforms('cifar10')[0], download=download_mapping['cifar10']),
        'cifar100': None if download_mapping['cifar100'] == False else CIFAR100Pair(root='data', train=True, transform=get_transforms('cifar100')[0], download=download_mapping['cifar100']),
        'stl10': None if download_mapping['stl10'] == False else STL10Pair(root='data', split='unlabeled', transform=get_transforms('stl10')[0], download=download_mapping['stl10']),
        'imagenet10': None if download_mapping['imagenet10'] == False else filter_ImageNet(ImageNetPair('/shared/sets/datasets/vision/ImageNet', split='train', transform=get_transforms('imagenet10')[0]), name), 
        'imagenetdogs': None if download_mapping['imagenetdogs'] == False else filter_ImageNet(ImageNetPair('/shared/sets/datasets/vision/ImageNet', split='train', transform=get_transforms('imagenetdogs')[0]), name), 

    }
    
    test_data = {
        'cifar10': None if download_mapping['cifar10'] == False else CIFAR10Pair(root='data', train=False, transform=get_transforms('cifar10')[1], download=download_mapping['cifar10']),
        'cifar100': None if download_mapping['cifar100'] == False else CIFAR100Pair(root='data', train=False, transform=get_transforms('cifar100')[1], download=download_mapping['cifar100']),
        'stl10': None if download_mapping['stl10'] == False else STL10Pair(root='data', split='test', transform=get_transforms('stl10')[1], download=download_mapping['stl10']),
        'imagenet10': None if download_mapping['imagenet10'] == False else filter_ImageNet(ImageNetPair('/shared/sets/datasets/vision/ImageNet', split='val', transform=get_transforms('imagenet10')[1]), name),
        'imagenetdogs': None if download_mapping['imagenetdogs'] == False else filter_ImageNet(ImageNetPair('/shared/sets/datasets/vision/ImageNet', split='val', transform=get_transforms('imagenetdogs')[1]), name),

    }

    memory_data = {
        'cifar10': None if download_mapping['cifar10'] == False else CIFAR10Pair(root='data', train=True, transform=get_transforms('cifar10')[1], download=download_mapping['cifar10']),
        'cifar100': None if download_mapping['cifar100'] == False else CIFAR100Pair(root='data', train=True, transform=get_transforms('cifar100')[1], download=download_mapping['cifar100']),
        'stl10': None if download_mapping['stl10'] == False else STL10Pair(root='data', split='train', transform=get_transforms('stl10')[1], download=download_mapping['stl10']),
        'imagenet10': None if download_mapping['imagenet10'] == False else filter_ImageNet(ImageNetPair('/shared/sets/datasets/vision/ImageNet', split='train', transform=get_transforms('imagenet10')[1]), name),
        'imagenetdogs': None if download_mapping['imagenetdogs'] == False else filter_ImageNet(ImageNetPair('/shared/sets/datasets/vision/ImageNet', split='train', transform=get_transforms('imagenetdogs')[1]), name),

    }
    return train_data[name], memory_data[name], test_data[name]


def filter_ImageNet(image_dataset, name):
    if name == 'imagenet10':
            subset_winds = [
                "n02056570",
                "n02085936",
                "n02128757",
                "n02690373",
                "n02692877",
                "n03095699",
                "n04254680",
                "n04285008",
                "n04467665",
                "n07747607"
            ]
    elif name == 'imagenetdogs':
            subset_winds = [
                "n02085936",
                "n02086646",
                "n02088238",
                "n02091467",
                "n02097130",
                "n02099601",
                "n02101388",
                "n02101556",
                "n02102177",
                "n02105056",
                "n02105412",
                "n02105855",
                "n02107142",
                "n02110958",
                "n02112137"
            ]
    subset_idx = [idx for idx, target in enumerate(image_dataset.wnids) if target in subset_winds]
    subset_indices = [idx for idx, target in enumerate(image_dataset.targets) if target in subset_idx]
    class_names = image_dataset.classes
    classes = tuple([class_names[c][0] for c in subset_idx])
    image_dataset = Subset(image_dataset, subset_indices)
    image_dataset.subset_index_attr = subset_idx
    return image_dataset

class ImageNetPair(ImageNet):
    """ImageNet Dataset.
    """

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return pos_1, pos_2, target



class STL10Pair(STL10):
    """STL10 Dataset.
    """

    def __getitem__(self, index):
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None        

        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target



class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target



class CIFAR100Pair(CIFAR100):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target
    

def reassing_classes(dataset, name):
    if name == 'cifar100':
        cifar100superclass = ["aquatic mammals","fish","flowers", "food containers", "fruit and vegetables", "household electrical devices", 
                            "household furniture", "insects", "large carnivores", "large man-made outdoor things", 
                            "large natural outdoor scenes", "large omnivores and herbivores", "medium-sized mammals",
                            "non-insect invertebrates", "people", "reptiles", "small mammals", "trees", "vehicles 1", "vehicles 2"]
        cifar100class = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                            ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                            ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                            ['bottle', 'bowl', 'can', 'cup', 'plate'],
                            ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                            ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                            ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                            ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                            ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                            ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                            ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                            ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                            ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                            ['crab', 'lobster', 'snail', 'spider', 'worm'],
                            ['baby', 'boy', 'girl', 'man', 'woman'],
                            ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                            ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                            ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                            ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                            ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]
        cifar100dict = {dataset.class_to_idx[c]: i for i, sc in enumerate(cifar100superclass) for c in cifar100class[i]}
        for i, t in enumerate(dataset.targets):
            dataset.targets[i] = cifar100dict[dataset.targets[i]]
    return dataset

def concat_datasets(train, test, name):
    if name == 'cifar10' or name =='cifar100' or name == 'stl10':
        return ConcatDataset([train, test])
    if name == 'imagenet10' or name == 'imagenetdogs':
        return train