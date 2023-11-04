import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm


class VQSMPLMotionDataset(data.Dataset):
    def __init__(self, dataset_name, window_size=64, unit_length=4):
        self.window_size = window_size
        self.unit_length = unit_length

        self.data_dict = np.load('/mnt/disk_2/jinpeng/t2m-gpt/dataset/debug/data_dict_p2m.npy', allow_pickle=True).item()
        self.name_list = list(self.data_dict.keys())

        # print("Total number of motions {}".format(len(self.data)))

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        name = self.name_list[item]
        features = self.data_dict[name]['motion']['features']
        features = features.to(torch.float32)
        return features


def DATALoader(dataset_name,
               batch_size,
               num_workers=8,
               window_size=64,
               unit_length=4):
    trainSet = VQSMPLMotionDataset(dataset_name, window_size=window_size, unit_length=unit_length)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                               batch_size,
                                               shuffle=True,
                                               # sampler=sampler,
                                               num_workers=num_workers,
                                               # collate_fn=collate_fn,
                                               drop_last=True)
    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x