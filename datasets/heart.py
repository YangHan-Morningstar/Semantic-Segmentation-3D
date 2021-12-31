from torch.utils.data import Dataset
import numpy as np


class Heart(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self.base_dir = base_dir
        self.transform = transform
        self.split = split
        self.sample_list = []
        if split == 'train':
            self.image_ids = [i for i in range(1, 21)]
        else:
            self.image_ids = [i for i in range(1, 41)]
        if num is not None:
            self.image_list = self.image_ids[: num]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_id = '0{}'.format(image_id) if image_id <= 9 else str(image_id)

        image = np.load('{}/train/{}_volumes.npy'.format(self.base_dir, image_id))[0]
        label = np.load('{}/train/{}_label.npy'.format(self.base_dir, image_id))
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
