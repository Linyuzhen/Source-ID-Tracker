import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from noise_layers.crop import random_float

class Resize(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """
    def __init__(self, resize_ratio_min, resize_ratio_max, interpolation_method='bilinear'):
        super(Resize, self).__init__()
        self.resize_ratio_min = resize_ratio_min
        self.resize_ratio_max = resize_ratio_max
        self.interpolation_method = interpolation_method


    def forward(self, container, secret_image, face_mask):

        resize_ratio = random_float(self.resize_ratio_min, self.resize_ratio_max)
        container_resize = F.interpolate(
                                    container,
                                    scale_factor=(resize_ratio, resize_ratio),
                                    mode=self.interpolation_method)
        secret_resize = F.interpolate(
                                    secret_image,
                                    scale_factor=(resize_ratio, resize_ratio),
                                    mode=self.interpolation_method)
        face_mask_resize = F.interpolate(
                                    face_mask,
                                    scale_factor=(resize_ratio, resize_ratio),
                                    mode=self.interpolation_method)
        return container_resize, secret_resize, face_mask_resize
