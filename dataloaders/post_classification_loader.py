import torch
import os
import numpy as np
from torch.utils.data import Dataset
from utils import oversample_cloud


class PostClassificationDataset(Dataset):
    def __init__(self,load_path,oversample= 2000):
        self.data_dict = torch.load(load_path)
        self.class_labels = ['nochange','removed',"added",'change',"color_change"]
        self.class_int_dict = {x:self.class_labels.index(x) for x in self.class_labels}
        self.oversample= oversample

    def __len__(self):
        return len(self.data_dict)

    def get_weights(self):
        counts = [0]*len(self.class_int_dict)
        for index,data_dict in self.data_dict.items():
            label = self.class_int_dict[data_dict['class']]
            counts[label]+=1
        return (torch.Tensor(counts)/sum(counts))**-1

    def __getitem__(self, idx):
        index_dict = self.data_dict[idx]
        extract_0,extract_1,label = index_dict[0],index_dict[1],index_dict['class']
        return oversample_cloud(extract_0,self.oversample)[:self.oversample,...],oversample_cloud(extract_1,self.oversample)[:self.oversample,...],torch.LongTensor([self.class_int_dict[label]])