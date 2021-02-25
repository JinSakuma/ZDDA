import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob


def set_paths(root, dataset_name, phase):
    keys = ['{}'.format(int(i)) for i in range(10)]
    values = [[] for _ in range(10)]
    path_dict = dict(zip(keys, values))
    for i in range(10):
        imgs_path = sorted(glob.glob(os.path.join(root, dataset_name, phase, '{}'.format(int(i)), '*.png')))
        path_dict['{}'.format(int(i))] += imgs_path
        
    return path_dict


def make_abcd_dataset(source_dict, target_dict, d_list=[5, 6, 7, 8, 9], max_num=5000, cls_flg=False):
    X_a, y_a, X_b, y_b, X_c, y_c, X_d, y_d = [], [], [], [], [], [], [], []
    src_list = list(source_dict.values())
    for i, s in enumerate(src_list):
        if not i in d_list: 
            X_a.extend(s[:max_num])
            y_a.extend([i for _ in range(max_num)])
        else:
            X_b.extend(s[:max_num])
            if cls_flg:
                y_b.extend([i-5 for _ in range(max_num)])
            else:
                y_b.extend([i for _ in range(max_num)])
            
        
    tgt_list = list(target_dict.values())
    for i, t in enumerate(tgt_list):
        if not i in d_list: 
            X_c.extend(t[:max_num])
            y_c.extend([i for _ in range(max_num)])
        else:
            X_d.extend(t[:max_num])
            if cls_flg:
                y_d.extend([i-5 for _ in range(max_num)])
            else:
                y_d.extend([i for _ in range(max_num)])
            
    return (np.asarray(X_a), np.asarray(y_a)), (np.asarray(X_b), np.asarray(y_b)), (np.asarray(X_c), np.asarray(y_c)), (np.asarray(X_d), np.asarray(y_d))


class MyDataset(Dataset):
    def __init__(self, path, label, domain, transform):
        assert len(path) == len(label)
        self.image_path = path
        self.label = torch.LongTensor(label)
        self.domain = torch.LongTensor(domain)
        self.transform = transform
        
    def __getitem__(self, index):
        path = self.image_path[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), self.label[index], self.domain[index]

    def __len__(self):
        return len(self.image_path)
    

class MyDataLoader(Dataset):
    def __init__(self, dataset, shuffle=True, batch_size=1):
        super().__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.order = []
        if shuffle:
            self.order = np.random.permutation(len(self.dataset))
        else:
            self.order = np.arange(len(self.dataset))
        self.curr_idx = -1

    def __len__(self):
        return int(len(self.dataset)/self.batch_size)

    def __getitem__(self, idx):

        jdx = self.order[idx*self.batch_size:(idx+1)*self.batch_size]

        X_list, y_list, d_list = [], [], []
        for i in range(self.batch_size):
            X, y, d = self.dataset[jdx[i]]
            
            X_list.append(X)
            y_list.append(y)
            d_list.append(d)

        return torch.stack(X_list), torch.stack(y_list), torch.stack(d_list)

    def next(self):
        self.curr_idx += 1
        if self.curr_idx >= self.__len__():
            self.curr_idx = 0
        return self.__getitem__(self.curr_idx)

    def on_epoch_end(self):
        self.order = np.random.permutation(len(self.dataset))

        
class PairDataLoader(Dataset):
    def __init__(self, dataset1, dataset2, shuffle=True, batch_size=1):
        super().__init__()

        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.batch_size = batch_size
        self.order = []
        if shuffle:
            self.order = np.random.permutation(len(self.dataset1))
        else:
            self.order = np.arange(len(self.dataset1))
        self.curr_idx = -1

    def __len__(self):
        return int(len(self.dataset1)/self.batch_size)

    def __getitem__(self, idx):

        jdx = self.order[idx*self.batch_size:(idx+1)*self.batch_size]

        X_list1, y_list1, d_list1 = [], [], []
        X_list2, y_list2, d_list2 = [], [], []
        for i in range(self.batch_size):
            X1, y1, d1 = self.dataset1[jdx[i]]
            X2, y2, d2 = self.dataset2[jdx[i]]
            
            X_list1.append(X1)
            y_list1.append(y1)
            d_list1.append(d1)
            X_list2.append(X2)
            y_list2.append(y2)
            d_list2.append(d2)

        return (torch.stack(X_list1), torch.stack(y_list1), torch.stack(d_list1)), (torch.stack(X_list2), torch.stack(y_list2), torch.stack(d_list2))

    def next(self):
        self.curr_idx += 1
        if self.curr_idx >= self.__len__():
            self.curr_idx = 0
        return self.__getitem__(self.curr_idx)

    def on_epoch_end(self):
        self.order = np.random.permutation(len(self.dataset1))