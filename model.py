import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import utils


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', is_bn=False, strides=1):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2))
        self.is_bn = is_bn
        if self.is_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        # default: using he_normal as the kernel initializer
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.is_bn:
            outputs = self.bn(outputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
            else:
                raise NotImplementedError
        return outputs


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', is_bn=False, strides=1):
        super(BasicBlock, self).__init__()
        self.is_bn = is_bn

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=kernel_size, activation=None, is_bn=False,
                            strides=strides)
        if self.is_bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = Conv2D(in_channels, out_channels, kernel_size=kernel_size, activation=None, is_bn=False, strides=1)
        # if self.is_bn:
        # self.bn2 = nn.BatchNorm2d(out_channels)
        if self.is_bn:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides, bias=False)
        self.strides = strides

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.is_bn:
            out = self.bn1(out)
        # out = self.relu(out)

        # out = self.conv2(out)
        # if self.is_bn:
        # out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# Some codes borrowed form
# https://github.com/SeuTao/TGS-Salt-Identification-Challenge-2018-_4th_place_solution/model/model.py
class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel // reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        chn_se = torch.mul(x, chn_se)

        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        # Preparation Network
        self.pad = (0, 1, 0, 1)
        self.conv1 = nn.Conv2d(3, 15, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 10, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(3, 5, kernel_size=7, padding=3)
        self.conv4 = nn.Conv2d(30, 15, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(30, 10, kernel_size=5, padding=2)
        self.conv6 = nn.Conv2d(30, 5, kernel_size=7, padding=3)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x4 = torch.cat([x1, x2, x3], axis=1)
        x1 = F.relu(self.conv4(x4))
        x2 = F.relu(self.conv5(x4))
        x3 = F.relu(self.conv6(x4))
        x4 = torch.cat([x1, x2, x3], axis=1)
        return x4


class ENet(nn.Module):
    def __init__(self):
        super(ENet, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.PNet = PNet()
        self.conv1 = Conv2D(33, 32, 3, activation='relu', is_bn=False)
        self.conv2 = BasicBlock(32, 32, 3, activation='relu', is_bn=False, strides=2)
        self.conv3 = BasicBlock(32, 64, 3, activation='relu', is_bn=False, strides=2)
        self.conv4 = BasicBlock(64, 128, 3, activation='relu', is_bn=False, strides=2)
        self.conv5 = BasicBlock(128, 256, 3, activation='relu', is_bn=False, strides=2)
        self.up6 = Conv2D(256, 128, 3, activation='relu', is_bn=False)
        self.conv6 = Conv2D(256, 128, 3, activation='relu', is_bn=False)
        self.SCSE6 = SCSEBlock(128)
        self.up7 = Conv2D(128, 64, 3, activation='relu', is_bn=False)
        self.conv7 = Conv2D(128, 64, 3, activation='relu', is_bn=False)
        self.SCSE7 = SCSEBlock(64)
        self.up8 = Conv2D(64, 32, 3, activation='relu', is_bn=False)
        self.conv8 = Conv2D(64, 32, 3, activation='relu', is_bn=False)
        self.SCSE8 = SCSEBlock(32)
        self.up9 = Conv2D(32, 32, 3, activation='relu', is_bn=False)
        self.conv9 = Conv2D(97, 32, 3, activation='relu', is_bn=False)
        self.SCSE9 = SCSEBlock(32)
        self.residual = Conv2D(32, 3, 1, activation=None, is_bn=False)

    def forward(self, source_image, fake_image, face_mask):
        source_image_fea = self.PNet(source_image)
        inputs = torch.cat([fake_image, source_image_fea], dim=1)
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        up6 = self.up6(nn.Upsample(scale_factor=(2, 2))(conv5))
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = self.conv6(merge6)
        conv6 = self.SCSE6(conv6)
        up7 = self.up7(nn.Upsample(scale_factor=(2, 2))(conv6))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.conv7(merge7)
        conv7 = self.SCSE7(conv7)
        up8 = self.up8(nn.Upsample(scale_factor=(2, 2))(conv7))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = self.conv8(merge8)
        conv8 = self.SCSE8(conv8)
        up9 = self.up9(nn.Upsample(scale_factor=(2, 2))(conv8))
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = self.conv9(merge9)
        conv9 = self.SCSE9(conv9)
        residual = self.sigmoid(self.residual(conv9))
        # residual = residual * face_mask + source_image * (1.0 - face_mask)
        residual = residual * face_mask + fake_image * (1.0 - face_mask)
        return residual


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.decoder = nn.Sequential(
            Conv2D(3, 32, 3, strides=1, activation='relu', is_bn=False),
            Conv2D(32, 64, 3, strides=1, activation='relu', is_bn=False),
            Conv2D(64, 128, 3, strides=1, activation='relu', is_bn=False),
            Conv2D(128, 64, 3, strides=1, activation='relu', is_bn=False),
            Conv2D(64, 32, 3, strides=1, activation='relu', is_bn=False),
            Conv2D(32, 3, 3, strides=1, activation=None, is_bn=False))

    def forward(self, image, face_mask):
        # decoder_image = torch.tanh(self.decoder(image))
        decoder_image = self.sigmoid(self.decoder(image))
        image = decoder_image * face_mask + image * (1.0 - face_mask)
        return image


def dsl_net(encoded_image, secret_image, face_mask, args, global_step, Crop_layer, Resize_layer):
    encoded_image = encoded_image.cpu()
    # encoded_image = (encoded_image + 1.0)/2.0
    # secret_image = (secret_image + 1.0)/2.0
    sh = encoded_image.size()
    ramp_fn = lambda ramp: np.min([global_step / ramp, 1.])

    rnd_bri = ramp_fn(args.rnd_bri_ramp) * args.rnd_bri
    rnd_hue = ramp_fn(args.rnd_hue_ramp) * args.rnd_hue
    rnd_brightness = utils.get_rnd_brightness_torch(rnd_bri, rnd_hue, args.batch_size)  # [batch_size, 3, 1, 1]
    jpeg_quality = 100. - torch.rand(1)[0] * ramp_fn(args.jpeg_quality_ramp) * (100. - args.jpeg_quality)
    rnd_noise = torch.rand(1)[0] * ramp_fn(args.rnd_noise_ramp) * args.rnd_noise

    contrast_low = 1. - (1. - args.contrast_low) * ramp_fn(args.contrast_ramp)
    contrast_high = 1. + (args.contrast_high - 1.) * ramp_fn(args.contrast_ramp)
    contrast_params = [contrast_low, contrast_high]

    rnd_sat = torch.rand(1)[0] * ramp_fn(args.rnd_sat_ramp) * args.rnd_sat

    # Resize
    resize_ratio_min = 1. - args.rnd_resize * ramp_fn(args.rnd_resize_ramp)
    resize_ratio_max = 1. + args.rnd_resize * ramp_fn(args.rnd_resize_ramp)
    Resize_layer.resize_ratio_min = resize_ratio_min
    Resize_layer.resize_ratio_max = resize_ratio_max
    encoded_image, secret_image, face_mask = Resize_layer(encoded_image, secret_image, face_mask)

    # Crop
    ratio_range = 1. - args.rnd_crop * ramp_fn(args.rnd_crop_ramp)
    ratio_range = 1. - args.rnd_crop * ramp_fn(args.rnd_crop_ramp)
    Crop_layer.height_ratio_range = [ratio_range, 1.0]
    Crop_layer.width_ratio_range = [ratio_range, 1.0]
    encoded_image, secret_image, face_mask = Crop_layer(encoded_image, secret_image, face_mask)

    # Resize back to 252x256
    encoded_image = F.interpolate(encoded_image, size=(args.resolution, args.resolution), mode='bilinear')
    secret_image = F.interpolate(secret_image, size=(args.resolution, args.resolution), mode='bilinear')
    face_mask = F.interpolate(face_mask, size=(args.resolution, args.resolution), mode='bilinear')
    face_mask[face_mask > 0.5] = 1.0

    # blur the code borrowed from https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
    kernel_size = 5
    dim = 2
    kernel_size = [kernel_size] * dim
    sigma = np.random.randint(2, 5, size=1)[0]
    sigma = [sigma] * dim

    # The gaussian kernel is the product of the
    # gaussian function of each dimension.
    kernel = 1
    meshgrids = torch.meshgrid(
        [
            torch.arange(size, dtype=torch.float32)
            for size in kernel_size
        ]
    )
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                  torch.exp(-((mgrid - mean) / std) ** 2 / 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(3, *[1] * (kernel.dim() - 1))

    # f = utils.random_blur_kernel(probs=[.25, .25], N_blur=kernel_size, sigrange_gauss=[1., 3.], sigrange_line=[.25, 1.], wmin_line=3)
    if args.is_cuda:
        kernel = kernel.cuda()

    if np.random.rand() < args.blur_prob:
        encoded_image = F.conv2d(encoded_image, kernel, bias=None, padding=int((kernel_size[0] - 1) / 2), groups=3)

    # noise
    noise = torch.normal(mean=0, std=rnd_noise, size=encoded_image.size(), dtype=torch.float32)
    if args.is_cuda:
        noise = noise.cuda()
    encoded_image = encoded_image + noise
    encoded_image = torch.clamp(encoded_image, 0, 1)

    # contrast & brightness
    # contrast_scale = torch.Tensor(encoded_image.size()[0]).uniform_(contrast_params[0], contrast_params[1])
    # contrast_scale = contrast_scale.reshape(encoded_image.size()[0], 1, 1, 1)
    # if args.is_cuda:
    # contrast_scale = contrast_scale.cuda()
    # rnd_brightness = rnd_brightness.cuda()
    # encoded_image = encoded_image * contrast_scale
    # encoded_image = encoded_image + rnd_brightness
    # encoded_image = torch.clamp(encoded_image, 0, 1)

    # saturation
    # sat_weight = torch.FloatTensor([.3, .6, .1]).reshape(1, 3, 1, 1)
    # if args.is_cuda:
    # sat_weight = sat_weight.cuda()
    # encoded_image_lum = torch.mean(encoded_image * sat_weight, dim=1).unsqueeze_(1)
    # encoded_image = (1 - rnd_sat) * encoded_image + rnd_sat * encoded_image_lum

    # jpeg
    encoded_image = encoded_image.reshape([-1, 3, args.resolution, args.resolution])
    if not args.no_jpeg:
        encoded_image = utils.jpeg_compress_decompress(encoded_image, args.is_cuda, rounding=utils.round_only_at_0,
                                                       quality=jpeg_quality)

    # encoded_image = (encoded_image - 0.5) * 2.0
    # secret_image = (secret_image - 0.5) * 2.0
    encoded_image = encoded_image.cuda()
    return encoded_image, secret_image, face_mask
