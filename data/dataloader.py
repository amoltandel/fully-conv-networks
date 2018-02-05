import os
import scipy.io as sio
import numpy as np
import skimage.io as imgio

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ToTensor(object):
    def __call__(self, sample):
        input = torch.from_numpy(sample['input'])
        input = input.permute(2, 0, 1)
        return {'input': input.unsqueeze(0).type(torch.FloatTensor),
                'target': torch.from_numpy(sample['target']).type(torch.LongTensor)}

class DatasetADK20(Dataset):
    def __init__(
            self,
            dataset_path='./',
            train=True,
            transform=transforms.Compose([ToTensor()]),
            ratio=1.0,
            resolution_reduction=1.0
        ):
        self.input_ext = '.jpg'
        self.target_ext = '_seg.png'
        self.dataset_path = dataset_path
        self.transform = transform
        self.ratio = ratio
        self.reduction = resolution_reduction
        index_file = sio.loadmat(
                os.path.join(self.dataset_path, 'ADE20K_2016_07_26/index_ade20k.mat')
            )

        self.class_indices = {}
        for idx, info in enumerate(index_file['index'][0][0][6][0].tolist()):
            self.class_indices[idx] = info.tolist()

        self.num_classes = len(self.class_indices.values())

        file_list = []
        for each_file, each_folder in zip(
                index_file['index'][0][0][0][0],
                index_file['index'][0][0][1][0]
            ):
            path = os.path.join(
                    self.dataset_path,
                    each_folder.tolist()[0],
                    each_file.tolist()[0][:-4]
                )
            file_list.append(path)

        if train:
            string = 'Training'
            self.file_list = file_list[:20210]
        else:
            string = 'Validation'
            self.file_list = file_list[20210:]
        print(string + ' data loaded!')
    def __len__(self):
        return int(len(self.file_list) * self.ratio)

    def __getitem__(self, idx):
        img = self.file_list[idx]
        inp = imgio.imread(img + self.input_ext)
        seg_mask = imgio.imread(img + self.target_ext)
        r_comp  = seg_mask[:, :, 0]
        g_comp  = seg_mask[:, :, 1]

        target = r_comp / 10 * 256 + g_comp
        target = target.astype(int)
        sample = {'input': inp, 'target':target}
        if self.transform:
            a = self.transform(sample)
            w = a['input'].size(2)
            h = a['input'].size(3)
            a['input'] = a['input'][:, :, :int(self.reduction * w), :int(self.reduction * h)]
            a['target'] = a['target'][:int(self.reduction * w), :int(self.reduction * h)]
            return a
        raise Exception('not returning tensor')


if __name__ == '__main__':
    a = DatasetADK20()
    sample = a[0]
    print(a.file_list[0])
    print(sample['input'].size(), sample['target'].size())
    print(a.class_indices[447])
    a = DatasetADK20(resolution_reduction=0.5)
    sample = a[0]
    print(a.file_list[0])
    print(sample['input'].size(), sample['target'].size())
    print(a.class_indices[447])
