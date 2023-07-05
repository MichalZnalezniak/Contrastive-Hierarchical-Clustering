from model import Model
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
import argparse
import pandas as pd
import numpy
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from omegaconf import OmegaConf
import utils
from torch.utils.data import DataLoader
from tree_losses import probability_vec_with_level
from metrics import tree_acc
from thop import profile, clever_format
from tqdm import tqdm
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--dataset-name', default='cifar10', choices=['stl10', 'cifar10', 'cifar100', 'imagenet10', 'imagenetdogs'])
parser.add_argument('--save_point', default='./results/')

@torch.no_grad()
def eval():
    args = parser.parse_args()
    cfg = OmegaConf.load(f'cfg/{args.dataset_name}.yaml')

    writer = SummaryWriter(log_dir=f"./eval/{args.save_point.split('/')[-1]}")
    # Load .pth
    checkpoint = torch.load(args.save_point + "last_epoch_model.pth")
    model = Model(cfg=cfg).cuda()
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    print(model)
    model.load_state_dict(checkpoint)
    masks_for_level = torch.load(args.save_point + "last_epoch_model_masks.pth")
    print("Masks map")
    print(masks_for_level)
    model.masks_for_level = masks_for_level
    model.eval()
    # data prepare
    _ , memory_data, test_data = utils.get_contrastive_dataset(args.dataset_name)
    # CIFAR100 - Reassign classes
    memory_data = utils.reassing_classes(memory_data, args.dataset_name)
    test_data = utils.reassing_classes(test_data, args.dataset_name)
    dataset = utils.concat_datasets(memory_data, test_data, args.dataset_name)
    valid_loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=False)
    labels, predictions = [], []
    histograms_for_each_label_per_level = {cfg.tree.tree_level : numpy.array([numpy.zeros_like(torch.empty(2**cfg.tree.tree_level)) for i in range(0, cfg.dataset.number_classes)])}

    for data, _, target in tqdm(valid_loader):
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        _, _, tree_output = model(data)
        prob_features = probability_vec_with_level(tree_output, cfg.tree.tree_level)
        prob_features = model.masks_for_level[cfg.tree.tree_level] * prob_features
        for prediction, label in zip(torch.argmax(prob_features.detach(), dim=1), target.detach()):
            if hasattr(dataset, 'subset_index_attr'):
                label = torch.Tensor([dataset.subset_index_attr.index(label)]).to(dtype=torch.int64)
            predictions.append(prediction.item())
            labels.append(label.item())
            histograms_for_each_label_per_level[cfg.tree.tree_level][label.item()][prediction.item()] += 1
    df_cm = pd.DataFrame(histograms_for_each_label_per_level[cfg.tree.tree_level], index = [class1 for class1 in range(0,cfg.dataset.number_classes)], columns = [i for i in range(0,2**cfg.tree.tree_level)])
    nmi = normalized_mutual_info_score(labels, predictions)
    print(f' NMI {nmi}')
    ari = adjusted_rand_score(labels, predictions)
    print(f' ARI {ari}')
    tree_acc_val = tree_acc(df_cm)
    print(f' Tree accruacy {tree_acc_val}')

if __name__ == "__main__":
    eval()  
