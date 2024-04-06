import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50
from convnext import convnext_base, convnext_tiny


class Model(nn.Module):
    def __init__(self, cfg=None):
        super(Model, self).__init__()

        model_mapping = {
            'resnet18': resnet18(),
            'resnet34': resnet34(),
            'resnet50': resnet50(),
            'convnext': convnext_base()
        }

        final_fe_dim_mapping = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048,
            'convnext': 1024
        }
        model_arch = model_mapping['convnext']
        final_fe_dim = final_fe_dim_mapping['convnext']
        self.f = model_mapping['convnext']
        self.f.norm = torch.nn.Identity()
        self.f.head = torch.nn.Identity()
        # projection head
        self.g = nn.Sequential(nn.Linear(final_fe_dim, 512, bias=False), nn.BatchNorm1d(512),
                                nn.ReLU(inplace=True), nn.Linear(512, cfg.simclr.feature_dim_projection_head, bias=True))
        self.tree_model = nn.Sequential(nn.Linear(final_fe_dim_mapping['convnext'], ((2**(cfg.tree.tree_level+1))-1) - 2**cfg.tree.tree_level), nn.Sigmoid())
        self.masks_for_level = {level: torch.ones(2**level).cuda() for level in range(1, cfg.tree.tree_level+1)}


    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        tree_output = self.tree_model(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1), tree_output
