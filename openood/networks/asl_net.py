import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class ASLNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super(ASLNet, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        if hasattr(self.backbone, 'heads'):
            heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
            heads_layers["head"] = nn.Linear(self.backbone.feature_size, num_classes, bias=False)
            original_weight = self.backbone.heads[0].weight.clone()
            self.backbone.heads = nn.Sequential(heads_layers)
            self.backbone.heads[0].weight.data = original_weight
            self.classifier = self.backbone.heads  # 统一指向分类器
            
        elif hasattr(self.backbone, 'fc'):
            original_weight = self.backbone.fc.weight.clone()
            self.backbone.fc = nn.Linear(self.backbone.feature_size, num_classes, bias=False)
            self.backbone.fc.weight.data = original_weight.clone()
            self.classifier = self.backbone.fc  # 统一指向分类器

        self.register_buffer('scaling', torch.tensor(10))
        print('scaling', self.scaling)

    def set_scaling(self, scaling):
        if hasattr(self.backbone, 'fc'):
            device = self.backbone.fc.weight.device
        else:
            first_param = next(self.backbone.parameters())
            device = first_param.device
        self.tau = torch.tensor(scaling, device=device)
        print('tau', self.tau)

    def forward(self, x, return_feature=False, norm=True):
        _, feature = self.backbone(x, return_feature=True)
        
        if norm:
            feature = F.normalize(feature, dim=-1) * self.scaling
        
        output = self.classifier(feature)
        
        if return_feature:
            return output, feature
        return output