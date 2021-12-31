import torch
import numpy as np
import random
from medpy import metric


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_pretrained_weights(model_state_dict, pretrained_state_dict):
    '''
    modify details of loading code according to the difference architecture
    of pretrained model and yours
    '''
    for key in pretrained_state_dict.keys():
        if key in model_state_dict:
            model_state_dict[key] = pretrained_state_dict[key]
            print('params of {} have been loaded!'.format(key))
        else:
            print('loading params of {} failed!'.format(key))
    return model_state_dict


class SegmentationMetrics(object):
    def __init__(self, label_list):
        self.label_list = label_list
        self.metrics = ['Dice', 'Jaccard', '95HD', 'ASD']

    def initial_metric_dict(self):
        organs_metrics_dict = {}
        for i in range(1, len(self.label_list)):
            organs_metrics_dict[self.label_list[i]] = {
                'Dice': 0,
                'Jaccard': 0,
                '95HD': 0,
                'ASD': 0
            }
        organs_metrics_dict['mean'] = {
            'Dice': 0,
            'Jaccard': 0,
            '95HD': 0,
            'ASD': 0
        }
        return organs_metrics_dict

    def calculate_metric_in_single_case(self, pre, gt):
        organs_metric_dict = {}
        total_dice, total_jc, total_hd, total_asd = 0, 0, 0, 0
        for i in range(1, len(self.label_list)):
            pre_i = (pre == i).astype(np.float32)
            gt_i = (gt == i).astype(np.float32)
            dice_i = metric.binary.dc(pre_i, gt_i)
            jc_i = metric.binary.jc(pre_i, gt_i)
            hd_i = metric.binary.hd95(pre_i, gt_i)
            asd_i = metric.binary.asd(pre_i, gt_i)
            metric_dict = {
                'Dice': dice_i,
                'Jaccard': jc_i,
                '95HD': hd_i,
                'ASD': asd_i
            }
            organs_metric_dict[self.label_list[i]] = metric_dict

            total_dice += dice_i
            total_jc += jc_i
            total_hd += hd_i
            total_asd += asd_i

        organs_num = len(self.label_list) - 1
        organs_metric_dict['mean'] = {
            'Dice': total_dice / organs_num,
            'Jaccard': total_jc / organs_num,
            '95HD': total_hd / organs_num,
            'ASD': total_asd / organs_num
        }

        return organs_metric_dict
