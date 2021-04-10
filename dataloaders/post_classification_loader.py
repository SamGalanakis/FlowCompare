import torch
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import load_las, random_subsample,view_cloud_plotly,grid_split,co_min_max, circle_split,co_standardize,sep_standardize,unit_sphere,co_unit_sphere,extract_area
from torch.utils.data import Dataset



class PostClassificationDataset(Dataset):
    def __init__(self,load_path):
        self.data_dict = torch.load(load_path)
        self.class_labels = ['nochange','removed',"added",'change',"color_change"]
        self.class_int_dict = {x:self.class_labels.index(x) for x in self.class_labels}

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        index_dict = self.data_dict[idx]
        extract_0,extract_1,label = index_dict[0],index_dict[1],index_dict['class']
        return extract_0,extract_1,self.class_int_dict[label]