import os
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tensorboardX import SummaryWriter

from models.vnet import VNet
from datasets.heart import Heart
from datasets.utils import RandomCrop, ToTensor
from utils import setup_seed, load_pretrained_weights
from losses.dice_loss import SquareDiceLoss
from test import test_during_train


def train_per_epoch(model, train_dataloader, ce_loss_fn, dice_loss_fn, optimizer, writer):
    model.train()
    train_mean_loss = 0
    train_mean_loss_seg_dice = 0
    train_mean_loss_seg_ce = 0
    for i, sample_batch in enumerate(train_dataloader):
        volume_batch, label_batch = sample_batch['image'].cuda(), sample_batch['label'].cuda()

        # forward
        seg_out = model(volume_batch)

        # compute cross entropy loss
        loss_seg_ce = ce_loss_fn(seg_out, label_batch)

        # compute dice loss
        seg_out_soft = F.softmax(seg_out, dim=1)
        loss_seg_dice = dice_loss_fn(seg_out_soft, label_batch)

        # full loss
        loss = loss_seg_ce + loss_seg_dice

        # calculate gradient and backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_mean_loss += loss.item()
        train_mean_loss_seg_ce += loss_seg_ce.item()
        train_mean_loss_seg_dice += loss_seg_dice.item()

    train_mean_loss /= len(train_dataloader)
    train_mean_loss_seg_ce /= len(train_dataloader)
    train_mean_loss_seg_dice /= len(train_dataloader)

    writer.add_scalar('loss/loss_seg_dice', train_mean_loss_seg_dice, epoch)
    writer.add_scalar('loss/loss_seg_ce', train_mean_loss_seg_ce, epoch)
    writer.add_scalar('loss/loss', train_mean_loss, epoch)

    return train_mean_loss, train_mean_loss_seg_ce, train_mean_loss_seg_dice


def test_per_epoch(args, model, label_list, patch_size, writer, best_val_dice):
    image_id_list = []
    for i in range(1, 21):
        filepath = '{}/test/{}'.format(args.source_dir, '0{}'.format(i) if i <= 9 else str(i))
        image_id_list.append(filepath)

    organs_metrics_dict = test_during_train(
        model,
        image_id_list,
        label_list,
        patch_size,
        stride_dim0=int(patch_size[0] / 2),
        stride_dim1=int(patch_size[1] / 2),
        stride_dim2=int(patch_size[2] / 2)
    )

    dice = organs_metrics_dict['mean']['Dice']
    jaccard = organs_metrics_dict['mean']['Jaccard']
    hd = organs_metrics_dict['mean']['95HD']
    asd = organs_metrics_dict['mean']['ASD']

    writer.add_scalar('mean_metrics/Dice', dice, epoch)
    writer.add_scalar('mean_metrics/Jaccard', jaccard, epoch)
    writer.add_scalar('mean_metrics/95HD', hd, epoch)
    writer.add_scalar('mean_metrics/ASD', asd, epoch)

    for i in range(1, len(label_list)):
        organ_name = label_list[i]
        organ_metric = organs_metrics_dict[organ_name]
        writer.add_scalar('{}_metrics/Dice'.format(organ_name), organ_metric['Dice'], epoch)
        writer.add_scalar('{}_metrics/Jaccard'.format(organ_name), organ_metric['Jaccard'], jaccard, epoch)
        writer.add_scalar('{}_metrics/95HD'.format(organ_name), organ_metric['95HD'], epoch)
        writer.add_scalar('{}_metrics/ASD'.format(organ_name), organ_metric['ASD'], epoch)

    # save state dict of model according to a rule
    if dice > best_val_dice:
        if best_val_dice != 0:
            os.system('rm {}/*.pth'.format(args.trained_weights_filepath))
        best_val_dice = round(dice, 4)
        save_mode_path = '{}/model_dice_{}.pth'.format(args.trained_weights_filepath, best_val_dice)
        torch.save(model.module.state_dict(), save_mode_path)

    return dice, jaccard, hd, asd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--source_dir', default='./data/WholeHeart')
    parser.add_argument('--label_dict_dir', default='./dicts/label_list.npz')
    parser.add_argument('--cuda', type=str, default='1')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--base_lr', type=float, default=1e-2)
    parser.add_argument('--log_dir', default='./log')
    parser.add_argument('--trained_weights_filepath', default='./trained_weights')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--pretrained_weights_filepath', type=str, default=None)
    parser.add_argument('--patch_size', type=str, default='96,128,128',
                        help='param in random crop or inference, e.g. 96,128,128')
    args = parser.parse_args()

    setup_seed(args.seed)  # setting random seed

    if os.path.exists(args.log_dir):
        os.system('rm -r {}'.format(args.log_dir))
    if os.path.exists(args.trained_weights_filepath):
        os.system('rm -r {}'.format(args.trained_weights_filepath))
    os.makedirs(args.log_dir)
    os.makedirs(args.trained_weights_filepath)

    patch_size = [int(i) for i in args.patch_size.split(',')]
    cuda = [int(i) for i in args.cuda.split(',')]
    label_list = np.load(args.label_dict_dir)['data']

    # setting device of gpu
    torch.cuda.set_device(cuda[0])

    # dataset and dataloader
    train_dataset = Heart(
        base_dir=args.source_dir,
        split='train',
        transform=transforms.Compose([
            RandomCrop(patch_size),
            ToTensor(),
        ])
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        pin_memory=True,
        drop_last=False
    )

    # create model and setting cuda
    model = VNet(n_channels=1, n_classes=len(label_list), has_dropout=True)
    if args.pretrained:
        assert args.pretrained_weights_filepath is not None, 'no pretrained weights filepath!'
        print('Loading pretrained weights!')
        pretrained_state_dict = torch.load(args.pretrained_weights_filepath)
        new_model_state_dict = load_pretrained_weights(model.state_dict(), pretrained_state_dict)
        model.load_state_dict(new_model_state_dict)
    else:
        print('Randomly initial params!')
    model.to(cuda[0])
    model = nn.DataParallel(model, cuda)

    # create loss functions and setting cuda
    ce_loss_fn = nn.CrossEntropyLoss().cuda()
    dice_loss_fn = SquareDiceLoss(num_class=len(label_list)).cuda()

    # create optimizer and scheduler of learning rate
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-5)

    # create writer of tensorboard to record
    writer = SummaryWriter(args.log_dir)

    # train or test
    best_val_dice = 0
    epoch_loop = tqdm(range(args.epoch))
    for epoch in epoch_loop:
        # training
        train_mean_loss, train_mean_loss_seg_ce, train_mean_loss_seg_dice = train_per_epoch(
            model, train_dataloader, ce_loss_fn, dice_loss_fn, optimizer, writer
        )

        # testing
        val_epoch_threshold = int(args.epoch * 2 / 3)
        if epoch > val_epoch_threshold:  # flag for beginning to val to avoid some RuntimeError
            dice, jaccard, hd, asd = test_per_epoch(
                args, model, label_list, patch_size, writer, best_val_dice
            )
        else:
            dice, jaccard, hd, asd = None, None, None, None

        # update learning rate
        scheduler.step(epoch + 1)

        # update postfix of tqdm
        if epoch <= val_epoch_threshold:
            epoch_loop.set_postfix(train_loss=train_mean_loss)
        else:
            epoch_loop.set_postfix(
                train_loss=train_mean_loss, val_dice=dice, val_jaccard=jaccard, val_95hd=hd, val_asd=asd
            )

    writer.close()

    print("Train and Val finished! The best val dice is {}.".format(best_val_dice))
