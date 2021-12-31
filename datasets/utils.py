import numpy as np
import torch


class RandomCrop(object):
    def __init__(self, patch_size, padding_value=0, padding=False):
        super(RandomCrop, self).__init__()
        self.patch_size = patch_size
        self.padding_value = padding_value
        self.padding = padding

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if (label.shape[0] <= self.patch_size[0] or label.shape[1] <= self.patch_size[1] or label.shape[2] <= self.patch_size[2]) and self.padding:
            pw = max((self.patch_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.patch_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.patch_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=self.padding_value)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=self.padding_value)

        (d, w, h) = image.shape
        d1 = np.random.randint(0, d - self.patch_size[0])
        w1 = np.random.randint(0, w - self.patch_size[1])
        h1 = np.random.randint(0, h - self.patch_size[2])

        label = label[d1: d1 + self.patch_size[0], w1: w1 + self.patch_size[1], h1: h1 + self.patch_size[2]]
        image = image[d1: d1 + self.patch_size[0], w1: w1 + self.patch_size[1], h1: h1 + self.patch_size[2]]

        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        super(CreateOnehotLabel, self).__init__()
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label, 'onehot_label': onehot_label}


class ToTensor(object):
    def __init__(self):
        super(ToTensor, self).__init__()

    def __call__(self, sample):
        image = sample['image']
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        if 'onehot_label' in sample:
            return {
                'image': torch.from_numpy(image),
                'label': torch.from_numpy(sample['label']).long(),
                'onehot_label': torch.from_numpy(sample['onehot_label']).long()
            }
        else:
            return {
                'image': torch.from_numpy(image),
                'label': torch.from_numpy(sample['label']).long()
            }
