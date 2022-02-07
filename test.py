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
        stride_dim0=16,
        stride_dim1=16,
        stride_dim2=16):
    model.eval()
    with torch.no_grad():
        avg_metric = test_all_case(
            model,
            test_image_list,
            label_list=label_list,
            patch_size=patch_size,
            stride_dim0=stride_dim0,
            stride_dim1=stride_dim1,
            stride_dim2=stride_dim2
        )
    return avg_metric


def test_all_case(
        model,
        test_image_list,
        label_list,
        patch_size,
        stride_dim0=16,
        stride_dim1=16,
        stride_dim2=16):
    '''
    :param model: model with pretrained weights
    :param test_image_list: test filepath, etc
    :param label_list: label dict(list form)
    :param patch_size: size of input, must be less than the size of image
    :param stride_dim2: step in 2 dim of image
    :param stride_dim1: step in 1 dim of image
    :param stride_dim0: step in 0 dim of image
    :return: metrics dict of whole test dataset
    '''
    seg_metrics = SegmentationMetrics(label_list)
    metrics = seg_metrics.metrics
    total_organs_metrics_dict = seg_metrics.initial_metric_dict()

    for image_path in test_image_list:

        image = np.load('{}_volumes.npy'.format(image_path))[0]
        label = np.load('{}_label.npy'.format(image_path))

        prediction, score_map = test_single_case(
            model, image, len(label_list), stride_dim0, stride_dim1, stride_dim2, patch_size
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


def test_single_case(model, image, num_class, stride_dim0, stride_dim1, stride_dim2, patch_size):
    '''
    :param model: model with pretrained weights
    :param image: (dim0, dim1, dim2)
    :param num_class: n_channel
    :param stride_dim2: step in 2 dim of image
    :param stride_dim1: step in 1 dim of image
    :param stride_dim0: step in 0 dim of image
    :param patch_size: size of input, must be less than the size of image
    :return: 3D segmentation prediction(not one-hot) and probability map
    '''
    img_dim0, img_dim1, img_dim2 = image.shape
    patch_dim0, patch_dim1, patch_dim2 = patch_size

    assert img_dim0 >= patch_dim0 and img_dim1 >= patch_dim1 and img_dim2 >= patch_dim2, 'size of image must be larger than patch_size'

    s_dim0 = math.ceil((img_dim0 - patch_dim0) / stride_dim0) + 1
    s_dim1 = math.ceil((img_dim1 - patch_dim1) / stride_dim1) + 1
    s_dim2 = math.ceil((img_dim2 - patch_dim2) / stride_dim2) + 1

    score_map = np.zeros((num_class, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for s0 in range(0, s_dim0):
        index_dim0 = min(stride_dim0 * s0, img_dim0 - patch_dim0)
        for s1 in range(0, s_dim1):
            index_dim1 = min(stride_dim1 * s1, img_dim1 - patch_dim1)
            for s2 in range(0, s_dim2):
                index_dim2 = min(stride_dim2 * s2, img_dim2 - patch_dim2)
                test_patch = image[index_dim0: index_dim0 + patch_dim0, index_dim1: index_dim1 + patch_dim1, index_dim2: index_dim2 + patch_dim2]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                seg_out = model(test_patch)
                s1 = F.softmax(seg_out, dim=1).cpu().data.numpy()[0, :, :, :, :]
                score_map[:, index_dim0: index_dim0 + patch_dim0, index_dim1: index_dim1 + patch_dim1, index_dim2: index_dim2 + patch_dim2] += s1
                cnt[index_dim0: index_dim0 + patch_dim0, index_dim1: index_dim1 + patch_dim1, index_dim2: index_dim2 + patch_dim2] += 1

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
        stride_dim0=int(patch_size[0] / 2),
        stride_dim1=int(patch_size[1] / 2),
        stride_dim2=int(patch_size[2] / 2)
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
