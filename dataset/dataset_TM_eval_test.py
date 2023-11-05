from torch.utils.data import Dataset
import numpy as np
import torch


class OOD_Dataset(Dataset):
    def __init__(self, motion_dir, motion_length):
        # self.motion_dir = motion_dir
        self.motion_length = motion_length
        # self.data_dict = []
        # self.length_list = []
        # self.name_list = []
        # motion_files = findAllFile(motion_dir)
        # for motion_file in motion_files:
        #     try:
        #         motion = np.load(motion_file, allow_pickle=True).item()
        #         if len(motion['trans']) < self.motion_length:
        #             continue
        #         motion = smpl_data_to_matrix_and_trans(motion)
        #         with cs.open(os.path.join(motion_file.replace('.npy', '.txt').replace('motion_data', 'semantic_labels')), 'r', encoding='utf-8',
        #                      errors='ignore') as f:
        #             caption = f.readline()
        #         self.data_dict.append({'motion': motion,
        #                                'caption': caption,
        #                                'length': len(motion['features'])})
        #         self.length_list.append(len(motion['features']))
        #         self.name_list.append(motion_file)
        #     except:
        #         pass
        self.data_dict = np.load('/mnt/disk_2/jinpeng/AvatarCLIP/AvatarAnimate/data/ood_data_dict_368.npy', allow_pickle=True)
        self.name_list = np.load('/mnt/disk_1/jinpeng/T2M/data/debug/ood_name_list_368.npy', allow_pickle=True)
        # self.length_list = np.load('data/debug/ood_length_list.npy', allow_pickle=True)

    def __len__(self):
        return len(self.name_list)

    def load_keyid(self, keyid):
        data = self.data_dict[keyid]
        motion, caption, m_length = data['motion'], data['caption'], data['length']
        features = motion['features']
        features = features.T.unsqueeze(1).to(torch.float32)
        return features, caption, keyid

    def __getitem__(self, idx):
        data = self.data_dict[idx]
        motion, caption, m_length = data['motion'], data['caption'], data['length']
        features = motion['features']
        features = features.T.unsqueeze(1).to(torch.float32)
        return features, caption