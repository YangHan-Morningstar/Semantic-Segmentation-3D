import argparse
import torch
import math
import numpy as np
import torch.nn.functional as F
import os

from models.vnet import VNet
from utils import SegmentationMetrics


def test_during_train(
        model,
        test_image_list,
        label_list,
        patch_size,
        stride_x=16,
        stride_y=16,
        stride_z=16):
    model.eval()
    with torch.no_grad():
        avg_metric = test_all_case(
            model,
            test_image_list,
            label_list=label_list,
            patch_size=patch_size,
            stride_x=stride_x,
            stride_y=stride_y,
            stride_z=stride_z
        )
    return avg_metric


def test_all_case(
        model,
        test_image_list,
        label_list,
        patch_size,
        stride_x=16,
        stride_y=16,
        stride_z=16):
    '''
    :param model: model with pretrained weights
    :param test_image_list: test filepath, etc
    :param label_list: label dict(list form)
    :param patch_size: size of input, must be less than the size of image
    :param stride_x: step in x dim
    :param stride_y: step in y dim
    :param stride_z: step in z dim
    :return: metrics dict of whole test dataset
    '''
    seg_metrics = SegmentationMetrics(label_list)
    metrics = seg_metrics.metrics
    total_organs_metrics_dict = seg_metrics.initial_metric_dict()

    for image_path in test_image_list:

        image = np.load('{}_volumes.npy'.format(image_path))[0]
        label = np.load('{}_label.npy'.format(image_path))

        prediction, score_map = test_single_case(
            model, image, len(label_list), stride_x, stride_y, stride_z, patch_size
        )

        if np.sum(prediction) == 0:
            single_metric = seg_metrics.initial_metric_dict()
        else:
            single_metric = seg_metrics.calculate_metric_in_single_case(
                prediction, label
            )

        for i in range(1, len(label_list)):
            for j in range(len(metrics)):
                total_organs_metrics_dict[label_list[i]][metrics[j]] += single_metric[label_list[i]][metrics[j]]
        for i in range(len(metrics)):
            total_organs_metrics_dict['mean'][metrics[i]] += single_metric['mean'][metrics[i]]

    for i in range(1, len(label_list)):
        for j in range(len(metrics)):
            total_organs_metrics_dict[label_list[i]][metrics[j]] /= len(test_image_list)
    for i in range(len(metrics)):
        total_organs_metrics_dict['mean'][metrics[i]] /= len(test_image_list)

    return total_organs_metrics_dict


def test_single_case(model, image, num_class, stride_x, stride_y, stride_z, patch_size):
    '''
    :param model: model with pretrained weights
    :param image: 3D
    :param num_class: n_channel
    :param stride_x: step in x dim
    :param stride_y: step in y dim
    :param stride_z: step in z dim
    :param patch_size: size of input, must be less than the size of image
    :return: 3D segmentation prediction(not one-hot) and probability map
    '''
    d, h, w = image.shape
    dp, hp, wp = patch_size

    assert d >= dp and h >= hp and w >= wp, 'size of image must be larger than patch_size'

    sz = math.ceil((d - dp) / stride_z) + 1
    sy = math.ceil((h - hp) / stride_y) + 1
    sx = math.ceil((w - wp) / stride_x) + 1

    score_map = np.zeros((num_class, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for z in range(0, sz):
        zs = min(stride_z * z, d - dp)
        for y in range(0, sy):
            ys = min(stride_y * y, h - hp)
            for x in range(0, sx):
                xs = min(stride_x * x, w - wp)
                test_patch = image[zs: zs + dp, ys: ys + hp, xs: xs + wp]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                seg_out = model(test_patch)
                y = F.softmax(seg_out, dim=1).cpu().data.numpy()[0, :, :, :, :]
                score_map[:, zs: zs + dp, ys: ys + hp, xs: xs + wp] += y
                cnt[zs: zs + dp, ys: ys + hp, xs: xs + wp] += 1

    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    return label_map, score_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default='./data/WholeHeart')
    parser.add_argument('--cuda', type=str, default='1', help='GPU to use')
    parser.add_argument('--trained_weights_filepath', type=str, default='model_loss_0.5349.pth')
    parser.add_argument('--patch_size', type=str, default='96,128,128',
                        help='param in random crop or inference, e.g. 96,128,128')
    parser.add_argument('--label_dict_dir', default='./dicts/label_list.npz')
    args = parser.parse_args()

    torch.cuda.set_device(torch.device('cuda:{}'.format(args.cuda)))

    image_id_list = []
    for i in range(1, 21):
        filepath = '{}/test/{}'.format(args.source_dir, '0{}'.format(i) if i <= 9 else str(i))
        image_id_list.append(filepath)

    label_list = np.load(args.label_dict_dir)['data']
    patch_size = [int(i) for i in args.patch_size.split(',')]

    model = VNet(n_channels=1, n_classes=len(label_list)).cuda()
    model.load_state_dict(
        torch.load(
            args.trained_weights_filepath,
            map_location='cuda:{}'.format(args.cuda)
        )
    )

    model.eval()
    organs_metrics = test_all_case(
        model,
        image_id_list,
        label_list,
        patch_size,
        stride_x=int(patch_size[2] / 2),
        stride_y=int(patch_size[1] / 2),
        stride_z=int(patch_size[0] / 2)
    )

    for i in range(1, len(label_list)):
        organ_name = label_list[i]
        metric = organs_metrics[organ_name]
        print('{} Dice: {:.4f} Jaccard: {:.4f} 95HD: {:.4f} ASD: {:.4f}'.format(
            organ_name,
            metric['Dice'],
            metric['Jaccard'],
            metric['95HD'],
            metric['ASD']
        ))
    metric = organs_metrics['mean']
    print('{} Dice: {:.4f} Jaccard: {:.4f} 95HD: {:.4f} ASD: {:.4f}'.format(
        'mean',
        metric['Dice'],
        metric['Jaccard'],
        metric['95HD'],
        metric['ASD']
    ))
