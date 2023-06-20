import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50


class Model(nn.Module):
    def __init__(self, cfg=None):
        super(Model, self).__init__()

        model_mapping = {
            'resnet18': resnet18(),
            'resnet34': resnet34(),
            'resnet50': resnet50()
        }

        final_fe_dim_mapping = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048
        }
        model_arch = model_mapping[cfg.model.name]
        final_fe_dim = final_fe_dim_mapping[cfg.model.name]
        self.f = []
        for name, module in model_arch.named_children():
            if cfg.dataset.dataset_name == 'mnist' or cfg.dataset.dataset_name == 'fmnist':
                if name == 'conv1':
                    module = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if cfg.model.receptive_field_incrased:
                if name == 'conv1':
                    module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                    self.f.append(module)
            else:
                if not isinstance(module, nn.Linear):
                    self.f.append(module)
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(final_fe_dim, 512, bias=False), nn.BatchNorm1d(512),
                                nn.ReLU(inplace=True), nn.Linear(512, cfg.simclr.feature_dim_projection_head, bias=True))
        self.tree_model = nn.Sequential(nn.Linear(final_fe_dim_mapping[cfg.model.name], ((2**(cfg.tree.tree_level+1))-1) - 2**cfg.tree.tree_level), nn.Sigmoid())
        self.masks_for_level = {level: torch.ones(2**level).cuda() for level in range(1, cfg.tree.tree_level+1)}


    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        tree_output = self.tree_model(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1), tree_output
