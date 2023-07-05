import argparse
import os

import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm
from tree_losses import probability_vec_with_level, tree_loss, regularization_loss
import utils
from model import Model
from metrics import tree_acc
import numpy
from sklearn.metrics import normalized_mutual_info_score
from torch.utils.tensorboard import SummaryWriter
import logging
from omegaconf import OmegaConf

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer, epoch, cfg):
    mean_of_probs_per_level_per_epoch = {level: torch.zeros(2**level).cuda() for level in range(1, cfg.tree.tree_level + 1)}
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    total_tree_loss, total_reg_loss, total_simclr_loss = 0.0, 0.0, 0.0
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, out_1, tree_output1 = net(pos_1)
        feature_2, out_2, tree_output2 = net(pos_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss_simclr = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        ##
        train_optimizer.zero_grad()
        if epoch > cfg.training.pretraining_epochs:
            tree_loss_value = tree_loss(tree_output1, tree_output2, batch_size, net.masks_for_level, mean_of_probs_per_level_per_epoch, cfg.tree.tree_level)
            regularization_loss_value = regularization_loss(tree_output1, tree_output2, net.masks_for_level, cfg.tree.tree_level)
            loss = loss_simclr + tree_loss_value + (2**(-cfg.tree.tree_level)*regularization_loss_value)
        else:
            loss = loss_simclr
            tree_loss_value = torch.zeros([1]) # Don't calculate the loss
            regularization_loss_value = torch.zeros([1]) # Don't calculate the loss 

        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_tree_loss += tree_loss_value.item() * batch_size
        total_reg_loss += regularization_loss_value.item() * batch_size
        total_simclr_loss += loss_simclr.item() * batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    if epoch > cfg.training.start_pruning_epochs and epoch <= cfg.training.start_pruning_epochs + cfg.training.leaves_to_prune:
        x = mean_of_probs_per_level_per_epoch[cfg.tree.tree_level]/ len(data_loader)
        x = x.double()
        test = torch.where(x > 0.0, x, 1.0) 
        net.masks_for_level[cfg.tree.tree_level][torch.argmin(test)] = 0
        print(net.masks_for_level[cfg.tree.tree_level])
    return total_loss / total_num, total_tree_loss / total_num, total_simclr_loss / total_num, total_reg_loss / total_num

# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader, epoch, cfg):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    histograms_for_each_label_per_level = {cfg.tree.tree_level : numpy.array([numpy.zeros_like(torch.empty(2**cfg.tree.tree_level)) for i in range(0, cfg.dataset.number_classes)])}
    labels, predictions = [], []
    with torch.no_grad():
        # generate feature bank
        if hasattr(memory_data_loader.dataset, 'targets'):
            for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
                feature, out, tree_output = net(data.cuda(non_blocking=True))
                feature_bank.append(feature)
            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            # [N]
            feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
            # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out, tree_output = net(data)
            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            if hasattr(memory_data_loader.dataset, 'targets'):
                
                c = len(memory_data.classes)

                sim_matrix = torch.mm(feature, feature_bank)
                # [B, K]
                sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
                # [B, K]
                sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
                sim_weight = (sim_weight / temperature).exp()

                # counts for each class
                one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
                # [B*K, C]
                one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
                # weighted score ---> [B, C]
                pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

                pred_labels = pred_scores.argsort(dim=-1, descending=True)
                total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            ## TREE PART
            prob_features = probability_vec_with_level(tree_output, cfg.tree.tree_level)
            prob_features = net.masks_for_level[cfg.tree.tree_level] * prob_features
            if hasattr(test_data_loader.dataset, 'subset_index_attr'):
                new_taget = []
                for elem in target:
                    new_taget.append(test_data_loader.dataset.subset_index_attr.index(elem))
                target = torch.Tensor(new_taget).to(dtype=torch.int64)
                
            for prediction, label in zip(torch.argmax(prob_features.detach(), dim=1), target.detach()):
                predictions.append(prediction.item())
                labels.append(label.item())
                histograms_for_each_label_per_level[cfg.tree.tree_level][label.item()][prediction.item()] += 1
            df_cm = pd.DataFrame(histograms_for_each_label_per_level[cfg.tree.tree_level], index = [class1 for class1 in range(0,cfg.dataset.number_classes)], columns = [i for i in range(0,2**cfg.tree.tree_level)])
            if epoch > 1000:
                tree_acc_val = tree_acc(df_cm)
            else:
                tree_acc_val = 0.0
            actuall_nmi = normalized_mutual_info_score(labels, predictions)
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}% Tree_acc:{:.2f} NMI:{:.2f}'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100, tree_acc_val, actuall_nmi))


    return total_top1 / total_num * 100, total_top5 / total_num * 100, tree_acc_val, actuall_nmi


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--dataset-name', default='cifar10', choices=['stl10', 'cifar10', 'cifar100', 'imagenet10', 'imagenetdogs'])

    args = parser.parse_args()
    cfg = OmegaConf.load(f'cfg/{args.dataset_name}.yaml')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k = cfg.simclr.feature_dim_projection_head, cfg.simclr.temperature, cfg.simclr.k
    batch_size, epochs = cfg.training.batch_size, cfg.training.epochs

    # data prepare
    train_data , memory_data, test_data = utils.get_contrastive_dataset(args.dataset_name)

    # train_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                              drop_last=True)
    # memory_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.test_transform, download=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    # test_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # logger
    writer = SummaryWriter()
    logging.basicConfig(filename=os.path.join(writer.log_dir, 'training.log'), level=logging.DEBUG)
    # model setup and optimizer config
    model = Model(cfg=cfg).cuda()


    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    print(model)
    # c = len(memory_data.classes)
    # training loop
# training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [],
                'tree_loss_train': [], 'reg_loss_train' : [], 'simclr_loss_train': [],
                 'tree_acc': [], 'nmi': []}
    save_name_pre = '{}_{}_{}_{}_{}'.format(feature_dim, temperature, k, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
    for epoch in range(1, epochs + 1):
        total_loss, tree_loss_train, reg_loss_train, simclr_loss_train = train(model, train_loader, optimizer, epoch, cfg)
        results['train_loss'].append(total_loss)
        results['tree_loss_train'].append(tree_loss_train)
        results['reg_loss_train'].append(reg_loss_train)
        results['simclr_loss_train'].append(simclr_loss_train)
        if epoch > cfg.training.pretraining_epochs:
            test_acc_1, test_acc_5, tree_acc_val, nmi = test(model, memory_loader, test_loader, epoch, cfg)
            results['test_acc@1'].append(test_acc_1)
            results['test_acc@5'].append(test_acc_5)
            results['tree_acc'].append(tree_acc_val)
            results['nmi'].append(nmi)
            writer.add_scalar('loss tree', tree_loss_train, global_step=epoch)
            writer.add_scalar('nmi', nmi, global_step=epoch)
            writer.add_scalar('tree_acc', tree_acc_val, global_step=epoch)
            # save statistics
            data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
            data_frame.to_csv(os.path.join(writer.log_dir, '{}_statistics.csv'.format(save_name_pre)), index_label='epoch')
        else:
            results['test_acc@1'].append(None)
            results['test_acc@5'].append(None)
            results['tree_acc'].append(None)
            results['nmi'].append(None)


    torch.save(model.state_dict(), 'results/{}_model.pth'.format('last_epoch'))
    torch.save(model.masks_for_level, 'results/last_epoch_model_masks.pth')
    
