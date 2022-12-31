"""
Author: HanChen
Date: 15.10.2020
"""
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse

import pandas as pd
from tqdm import tqdm
from kornia import color
import lpips

from model import ENet, RNet, dsl_net
from torchvision.utils import save_image
from transforms import build_transforms
from utils import AverageMeter
from ff_df import ff_df_Dataloader

from noise_layers.crop import Crop
from noise_layers.resize import Resize
from losses import Arcface_loss

import os


######################################################################
# Save model
def save_network(network, save_filename):
    torch.save(network.cpu().state_dict(), save_filename)
    if torch.cuda.is_available():
        network.cuda()


def load_network(network, save_filename):
    network.load_state_dict(torch.load(save_filename))
    return network


def parse_args():
    parser = argparse.ArgumentParser(description='Training network on ffdf dataset')

    parser.add_argument('--root_path', type=str, default='/data/linyz/FaceForensics_Fingerprints')
    parser.add_argument('--save_path', type=str, default='./save_results')
    # parser.add_argument('--pretrained', type=str, default='/home/linyz/UCL_Blending/xception-43020ad28.pth')     
    parser.add_argument('--face_id_save_path', type=str, default='/data/linyz/SIDT/Arcface.pth')

    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--rnd_bri_ramp', type=int, default=5000)
    parser.add_argument('--rnd_hue_ramp', type=int, default=5000)
    parser.add_argument('--jpeg_quality_ramp', type=float, default=5000)
    parser.add_argument('--rnd_noise_ramp', type=int, default=5000)
    parser.add_argument('--contrast_ramp', type=int, default=5000)
    parser.add_argument('--rnd_sat_ramp', type=int, default=5000)
    parser.add_argument('--rnd_crop_ramp', type=int, default=5000)  # Borrowed from HiDDeN
    parser.add_argument('--rnd_resize_ramp', type=int, default=5000)  # Borrowed from HiDDeN
    parser.add_argument('--rnd_bri', type=float, default=.1)
    parser.add_argument('--rnd_hue', type=float, default=.05)
    parser.add_argument('--jpeg_quality', type=float, default=50)
    parser.add_argument('--rnd_noise', type=float, default=.02)
    parser.add_argument('--contrast_low', type=float, default=.8)
    parser.add_argument('--contrast_high', type=float, default=1.2)
    parser.add_argument('--rnd_sat', type=float, default=0.5)
    parser.add_argument('--blur_prob', type=float, default=0.1)
    parser.add_argument('--no_jpeg', type=bool, default=False)
    parser.add_argument('--is_cuda', type=bool, default=False)
    parser.add_argument('--rnd_crop', type=float, default=0.2)  # Borrowed from HiDDeN
    parser.add_argument('--rnd_resize', type=float, default=0.2)  # Borrowed from HiDDeN

    parser.add_argument('--y_scale', type=float, default=1.0)
    parser.add_argument('--u_scale', type=float, default=10.0)
    parser.add_argument('--v_scale', type=float, default=10.0)

    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--val_epochs', type=int, default=10)
    parser.add_argument('--start_val_epochs', type=int, default=0)
    parser.add_argument('--adjust_lr_epochs', type=int, default=50)
    parser.add_argument('--base_lr', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--alpha', type=list, default=[1.0, 1.0, 1.0, 0.85, 0.85, 0.85],
                        help='alpha for L2 Loss of Container, CosSimilarity Loss of Container ID, Lpip,\
                             L2 Loss of Revealed, CosSimilarity Loss of Revealed ID, Lpip')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    transform_train, transform_test = build_transforms(args.resolution, args.resolution,
                                                       max_pixel_value=255.0, norm_mean=[0, 0, 0],
                                                       norm_std=[1.0, 1.0, 1.0])
    yuv_scales = torch.Tensor([args.y_scale, args.u_scale, args.v_scale]).cuda()

    with open('../train_df_fake_c23.txt', 'r') as f:
        fake_train_videos = f.readlines()
        fake_train_videos = [i.strip() for i in fake_train_videos]
    with open('../val_df_fake_c23.txt', 'r') as f:
        fake_val_videos = f.readlines()
        fake_val_videos = [i.strip() for i in fake_val_videos]

    with open('../train_df_source_c23.txt', 'r') as f:
        source_train_videos = f.readlines()
        source_train_videos = [i.strip() for i in source_train_videos]
    with open('../val_df_source_c23.txt', 'r') as f:
        source_val_videos = f.readlines()
        source_val_videos = [i.strip() for i in source_val_videos]

    train_dataset = ff_df_Dataloader(root_path=args.root_path, fake_video_names=fake_train_videos,
                                     source_video_names=source_train_videos,
                                     phase='train', transform=transform_train, size=(args.resolution, args.resolution))

    val_dataset = ff_df_Dataloader(root_path=args.root_path, fake_video_names=fake_val_videos,
                                   source_video_names=source_val_videos,
                                   phase='valid', transform=transform_test, test_frame_nums=30,
                                   size=(args.resolution, args.resolution))

    print('Test Images Number: %d' % len(val_dataset))
    print('All Train videos Number: %d' % (len(fake_train_videos)))
    print('Use Train videos Number: %d' % len(train_dataset))

    encoder = ENet().cuda()
    decoder = RNet().cuda()
    Crop_layer = Crop([1.0 - args.rnd_crop, 1.0], [1.0 - args.rnd_crop, 1.0])
    Resize_layer = Resize(1.0 - args.rnd_resize, 1.0 + args.rnd_resize)

    optimizerH = optim.Adam(filter(lambda p: p.requires_grad, 
                                   encoder.parameters()), lr=args.base_lr)
    optimizerR = optim.Adam(filter(lambda p: p.requires_grad, 
                                   decoder.parameters()), lr=args.base_lr)

    schedulerH = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerH, T_max=args.adjust_lr_epochs)
    schedulerR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerR, T_max=args.adjust_lr_epochs)

    cross_criterion = torch.nn.CrossEntropyLoss()
    # L1_criterion = nn.L1Loss().cuda()
    L2_criterion = nn.MSELoss().cuda()
    loss_fn_vgg = lpips.LPIPS(
        net='vgg').eval().cuda()  # closer to "traditional" perceptual loss, when used for optimization
    face_ID_criterion = Arcface_loss(pretrained=args.face_id_save_path).cuda()
    # binary_criterion = torch.nn.BCELoss()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True,
                                               shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=args.val_batch_size,
                                             drop_last=True, num_workers=4, pin_memory=True)

    best_loss = 1000.0
    global_step = 0
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)

        encoder.train(True)  # Set ENet to training mode
        decoder.train(True)  # Set RNet to training mode
        # Iterate over data (including images and labels).

        Hiding_l2losses = AverageMeter()
        Hiding_lpiplosses = AverageMeter()
        Hiding_idlosses = AverageMeter()
        Revealed_l2losses = AverageMeter()
        Revealed_lpiplosses = AverageMeter()
        Revealed_idlosses = AverageMeter()
        Sum_losses = AverageMeter()
        training_process = tqdm(train_loader)
        for idx, (fake_image, source_image, face_mask) in enumerate(training_process):
            if idx > 0:
                training_process.set_description(
                    "Epoch %d -- H_l2: %.4f, H_id: %.4f, H_lpip: %.4f, R_l2: %.4f, R_id: %.4f, R_lpip: %.4f, Sum_Loss: %.4f" %
                    (epoch, Hiding_l2losses.avg.item(), Hiding_idlosses.avg.item(), Hiding_lpiplosses.avg.item(),
                     Revealed_l2losses.avg.item(), Revealed_idlosses.avg.item(), Revealed_lpiplosses.avg.item(),
                     Sum_losses.avg.item()))
                # zero the parameter gradients
            optimizerH.zero_grad()
            optimizerR.zero_grad()

            # wrap them in Variable
            fake_image = Variable(fake_image.cuda().detach())
            source_image = Variable(source_image.cuda().detach())
            face_mask = Variable(face_mask.cuda().detach())

            container_fake_image = encoder(source_image, fake_image, face_mask)
            # lossH = L2_criterion(color.rgb_to_yuv(container_fake_image), color.rgb_to_yuv(fake_image))  # loss between cover and container
            lossH = torch.mean(((color.rgb_to_yuv(container_fake_image) - color.rgb_to_yuv(fake_image))) ** 2,
                               axis=[0, 2, 3])  # loss between cover and container
            lossH = torch.dot(lossH, yuv_scales)
            lpips_lossH = torch.mean(loss_fn_vgg((fake_image - 0.5) * 2.0, (container_fake_image - 0.5) * 2.0))
            face_ID_lossH = face_ID_criterion(fake_image, container_fake_image)
            Hiding_l2losses.update(lossH.cpu(), fake_image.size(0))
            Hiding_lpiplosses.update(lpips_lossH.cpu(), fake_image.size(0))
            Hiding_idlosses.update(face_ID_lossH.cpu(), fake_image.size(0))

            if epoch < 2:
                with torch.no_grad():
                    transformed_container_fake_image, transformed_source, transformed_face_mask = dsl_net(
                        container_fake_image,
                        source_image, face_mask,
                        args, global_step, Crop_layer, Resize_layer)  # dsl_net, data augmentation
                    rev_source_image = decoder(transformed_container_fake_image,
                                            transformed_face_mask)  # put concatenated image into R-net and get revealed secret image

                    # lossR = L2_criterion(color.rgb_to_yuv(rev_source_image), color.rgb_to_yuv(transformed_source))  # loss between secret image and revealed secret image
                    lossR = torch.mean(
                        ((color.rgb_to_yuv(rev_source_image) - color.rgb_to_yuv(transformed_source))) ** 2,
                        axis=[0, 2, 3])  # loss between cover and container
                    lossR = torch.dot(lossR, yuv_scales)
                    lpips_lossR = torch.mean(
                        loss_fn_vgg((transformed_source - 0.5) * 2.0, (rev_source_image - 0.5) * 2.0))
                    face_ID_lossR = face_ID_criterion(transformed_source, rev_source_image)
                    Revealed_l2losses.update(lossR.cpu(), fake_image.size(0))
                    Revealed_lpiplosses.update(lpips_lossR.cpu(), fake_image.size(0))
                    Revealed_idlosses.update(face_ID_lossR.cpu(), fake_image.size(0))

                loss = args.alpha[0] * lossH + args.alpha[1] * face_ID_lossH + args.alpha[2] * lpips_lossH

                # update the parameters
                loss.backward()
                optimizerH.step()
            else:
                transformed_container_fake_image, transformed_source, transformed_face_mask = dsl_net(
                    container_fake_image,
                    source_image, face_mask,
                    args, global_step, Crop_layer, Resize_layer)  # dsl_net, data augmentation
                rev_source_image = decoder(transformed_container_fake_image,
                                        transformed_face_mask)  # put concatenated image into R-net and get revealed secret image

                # lossR = L2_criterion(color.rgb_to_yuv(rev_source_image), color.rgb_to_yuv(transformed_source))  # loss between secret image and revealed secret image
                lossR = torch.mean(((color.rgb_to_yuv(rev_source_image) - color.rgb_to_yuv(transformed_source))) ** 2,
                                   axis=[0, 2, 3])  # loss between cover and container
                lossR = torch.dot(lossR, yuv_scales)
                lpips_lossR = torch.mean(loss_fn_vgg((transformed_source - 0.5) * 2.0, (rev_source_image - 0.5) * 2.0))
                face_ID_lossR = face_ID_criterion(transformed_source, rev_source_image)
                Revealed_l2losses.update(lossR.cpu(), fake_image.size(0))
                Revealed_lpiplosses.update(lpips_lossR.cpu(), fake_image.size(0))
                Revealed_idlosses.update(face_ID_lossR.cpu(), fake_image.size(0))
                loss = args.alpha[0] * lossH + args.alpha[1] * face_ID_lossH + args.alpha[2] * lpips_lossH \
                       + args.alpha[3] * lossR + args.alpha[4] * face_ID_lossR + args.alpha[5] * lpips_lossR
                # update the parameters
                loss.backward()
                optimizerH.step()
                optimizerR.step()

            Sum_losses.update(loss.cpu(), fake_image.size(0))
            global_step += 1

        if (epoch + 1) % args.val_epochs == 0:
            output_image = torch.cat((source_image.cpu(), fake_image.cpu(), container_fake_image.cpu(),
                                      transformed_container_fake_image.cpu(), transformed_source.cpu(),
                                      rev_source_image.cpu()), dim=0)
            save_image(output_image, '%s/images/output_train_%s.jpg' % (args.save_path, epoch),
                       normalize=True, nrow=args.batch_size)
            output_image = torch.cat((face_mask.cpu(), transformed_face_mask.cpu()), dim=0)
            save_image(output_image, '%s/images/output_mask.jpg' % args.save_path, normalize=True, nrow=args.batch_size)

        if (epoch + 1) % args.val_epochs == 0 and epoch > args.start_val_epochs:
            encoder.train(False)
            decoder.train(False)
            encoder.eval()
            decoder.eval()

            Hiding_l2losses = AverageMeter()
            Hiding_lpiplosses = AverageMeter()
            Hiding_idlosses = AverageMeter()
            Revealed_l2losses = AverageMeter()
            Revealed_lpiplosses = AverageMeter()
            Revealed_idlosses = AverageMeter()
            Sum_losses = AverageMeter()
            Sum_losses = AverageMeter()

            valid_process = tqdm(val_loader)
            for idx, (fake_image, source_image, face_mask) in enumerate(valid_process):
                if idx > 0:
                    valid_process.set_description(
                        "Epoch %d -- H_l2: %.4f, H_id: %.4f, H_lpip: %.4f, R_l2: %.4f, R_id: %.4f, R_lpip: %.4f, Sum_Loss: %.4f" %
                        (epoch, Hiding_l2losses.avg.item(), Hiding_idlosses.avg.item(), Hiding_lpiplosses.avg.item(),
                         Revealed_l2losses.avg.item(), Revealed_idlosses.avg.item(), Revealed_lpiplosses.avg.item(),
                         Sum_losses.avg.item()))

                    # wrap them in Variable
                fake_image = Variable(fake_image.cuda().detach())
                source_image = Variable(source_image.cuda().detach())
                face_mask = Variable(face_mask.cuda().detach())

                with torch.no_grad():
                    container_fake_image = encoder(source_image, fake_image, face_mask)
                    # lossH = L2_criterion(color.rgb_to_yuv(container_fake_image), color.rgb_to_yuv(fake_image))  # loss between cover and container
                    lossH = torch.mean(((color.rgb_to_yuv(container_fake_image) - color.rgb_to_yuv(fake_image))) ** 2,
                                       axis=[0, 2, 3])  # loss between cover and container
                    lossH = torch.dot(lossH, yuv_scales)
                    lpips_lossH = torch.mean(loss_fn_vgg((fake_image - 0.5) * 2.0, (container_fake_image - 0.5) * 2.0))

                    face_ID_lossH = face_ID_criterion(fake_image, container_fake_image)
                    Hiding_l2losses.update(lossH.cpu(), fake_image.size(0))
                    Hiding_lpiplosses.update(lpips_lossH.cpu(), fake_image.size(0))
                    Hiding_idlosses.update(face_ID_lossH.cpu(), fake_image.size(0))
                    rev_source_image = decoder(container_fake_image,
                                            face_mask)  # put concatenated image into R-net and get revealed secret image

                    # lossR = L2_criterion(color.rgb_to_yuv(rev_source_image), color.rgb_to_yuv(source_image))  # loss between secret image and revealed secret image
                    lossR = torch.mean(((color.rgb_to_yuv(rev_source_image) - color.rgb_to_yuv(source_image))) ** 2,
                                       axis=[0, 2, 3])  # loss between cover and container
                    lossR = torch.dot(lossR, yuv_scales)
                    lpips_lossR = torch.mean(loss_fn_vgg((source_image - 0.5) * 2.0, (rev_source_image - 0.5) * 2.0))
                    face_ID_lossR = face_ID_criterion(source_image, rev_source_image)
                    Revealed_l2losses.update(lossR.cpu(), fake_image.size(0))
                    Revealed_lpiplosses.update(lpips_lossR.cpu(), fake_image.size(0))
                    Revealed_idlosses.update(face_ID_lossR.cpu(), fake_image.size(0))
                    loss = args.alpha[0] * lossH + args.alpha[1] * face_ID_lossH + args.alpha[2] * lossR + args.alpha[
                        3] * face_ID_lossR
                    Sum_losses.update(loss.cpu(), fake_image.size(0))

            output_image = torch.cat((source_image.cpu(), fake_image.cpu(), face_mask.cpu(), container_fake_image.cpu(),
                                      rev_source_image.cpu()), dim=0)
            save_image(output_image, '%s/images/output_test_%s.jpg' % (args.save_path, epoch),
                       normalize=True, nrow=args.val_batch_size)

            df_acc = pd.DataFrame()
            df_acc['epoch'] = [epoch]
            df_acc['Hiding_l2losses'] = [Hiding_l2losses.avg.item()]
            df_acc['Hiding_lpiplosses'] = [Hiding_lpiplosses.avg.item()]
            df_acc['Hiding_idlosses'] = [Hiding_idlosses.avg.item()]
            df_acc['Revealed_l2losses'] = [Revealed_l2losses.avg.item()]
            df_acc['Revealed_lpiplosses'] = [Revealed_lpiplosses.avg.item()]
            df_acc['Revealed_idlosses'] = [Revealed_idlosses.avg.item()]
            df_acc['Sum_Loss'] = [Sum_losses.avg.item()]

            if epoch + 1 != (args.val_epochs + args.start_val_epochs):
                df_acc.to_csv('%s/report/validation.csv' % args.save_path, mode='a', index=None, header=None)
            else:
                df_acc.to_csv('%s/report/validation.csv' % args.save_path, mode='a', index=None)

            if best_loss > Sum_losses.avg.item():
                best_loss = Sum_losses.avg.item()
                save_network(encoder, '%s/models/encoder.pth' % args.save_path)
                save_network(decoder, '%s/models/decoder.pth' % args.save_path)

        if (epoch + 1) % 100 == 0:
            save_network(encoder, '%s/models/encoder_%s.pth' % (args.save_path, epoch))
            save_network(decoder, '%s/models/decoder_%s.pth' % (args.save_path, epoch))
        schedulerH.step()
        schedulerR.step()


if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists('%s/models' % args.save_path):
        os.makedirs('%s/models' % args.save_path)
    if not os.path.exists('%s/report' % args.save_path):
        os.makedirs('%s/report' % args.save_path)
    if not os.path.exists('%s/images' % args.save_path):
        os.makedirs('%s/images' % args.save_path)
    main()
