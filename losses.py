import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101
import sys


# Adapted from https://github.com/mindslab-ai/faceshifter
class Arcface_loss(nn.Module):
    def __init__(self, pretrained=None):
        super(Arcface_loss, self).__init__()

        self.net = resnet101(num_classes=256)
        self.net.eval()
        if pretrained is not None:
            # https://drive.google.com/file/d/1TAb6WNfusbL2Iv3tfRCpMXimZE9tnSUn/view?usp=sharing
            self.net.load_state_dict(torch.load(pretrained, map_location='cpu'))

        self.l1 = nn.L1Loss(reduction='mean')
        self.l2 = nn.MSELoss(reduction='mean')

    def id_loss(self, z_id_X, z_id_Y):
        inner_product = (torch.bmm(z_id_X.unsqueeze(1), z_id_Y.unsqueeze(2)).squeeze())
        # cosine_similarity = (torch.cosine_similarity(z_id_X, z_id_Y).squeeze())
        # print(inner_product)
        # print(inner_product.size())
        return self.l1(torch.ones_like(inner_product), inner_product)

    def forward(self, real_img, fake_img):
        # in the range [0.0, 1.0]
        # real_img_tmp = (real_img + 1.0)/2.0
        # fake_img_tmp = (fake_img + 1.0)/2.0
        z_id_X = self.net(F.interpolate(real_img, size=112, mode='bilinear'))
        z_id_X = F.normalize(z_id_X)
        z_id_X = z_id_X.detach()

        z_id_Y = self.net(F.interpolate(fake_img, size=112, mode='bilinear'))
        z_id_Y = F.normalize(z_id_Y)
        z_id_Y = z_id_Y.detach()

        Arcface_id_loss = self.id_loss(z_id_X, z_id_Y)
        return Arcface_id_loss