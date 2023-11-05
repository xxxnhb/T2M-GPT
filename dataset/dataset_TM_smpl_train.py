import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, feat_bias=5, unit_length=4, codebook_size=1024, tokenizer_name=None):
        self.max_length = 20
        self.pointer = 0
        self.data_root = '/mnt/disk_2/jinpeng/t2m-gpt/dataset/HumanML3D_SMPL'
        craft_data = np.load('/mnt/disk_2/jinpeng/t2m-gpt/dataset/debug/data_dict_p2m.npy', allow_pickle=True).item()
        data_dict = {}
        id_list = list(craft_data.keys())

        name_list = []
        for name in tqdm(id_list):
            try:
                m_token_list = np.load(pjoin(self.data_root, tokenizer_name, '%s.npy' % name))
                text_data = []
                with cs.open(pjoin('/mnt/disk_2/jinpeng/t2m-gpt/dataset/HumanML3D/texts', name + '.txt')) as f:
                    line = f.readline()
                    text_dict = {}
                    caption = line.strip().split('#')[0]

                    text_dict['caption'] = caption
                    text_data.append(text_dict)
                data_dict[name] = {'m_token_list': m_token_list,
                                   'text': text_data}
                name_list.append(name)
            except:
                # Some motion may not exist in KIT dataset
                pass

        self.data_dict = data_dict
        self.name_list = name_list

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        m_token_list, caption = data['m_token_list'], data['text'][0]['caption']
        m_tokens = random.choice(m_token_list)
        m_tokens_len = m_tokens.shape[0]

        return caption, m_tokens.reshape(-1), m_tokens_len


def DATALoader(dataset_name,
               batch_size, codebook_size, tokenizer_name, unit_length=4,
               num_workers=8):
    train_loader = torch.utils.data.DataLoader(
        Text2MotionDataset(dataset_name, codebook_size=codebook_size, tokenizer_name=tokenizer_name,
                           unit_length=unit_length),
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        # collate_fn=collate_fn,
        drop_last=True)

    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x