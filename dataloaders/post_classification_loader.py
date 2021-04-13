import torch
import os
import numpy as np
from torch.utils.data import Dataset
from utils import oversample_cloud


class PostClassificationDataset(Dataset):
    def __init__(self,load_path,oversample= 2000,keep_rgb=True,merge_classes=False):
        self.keep_rgb = keep_rgb
        self.data_dict = torch.load(load_path)
        self.class_labels = ['nochange','removed',"added",'change',"color_change"]
        
        self.class_int_dict = {x:self.class_labels.index(x) for x in self.class_labels}
        self.oversample= oversample
        self.merge_classes = merge_classes
       

    def __len__(self):
        return len(self.data_dict)
    
    def class_to_int(self,x):
        if not self.merge_classes:
            return self.class_int_dict[x]
        else:
            return 0 if x=='nochange' else 1
    
    def get_weights(self):
        if not self.merge_classes:
            counts = [0]*len(self.class_int_dict)
            for index,data_dict in self.data_dict.items():
                label = self.class_int_dict[data_dict['class']]
                counts[label]+=1
            return (torch.Tensor(counts)/sum(counts))**-1
        else:
            counter = 0
            for index,data_dict in self.data_dict.items():
                label = self.class_to_int(data_dict['class'])
                counter+=label
            return (torch.Tensor([(len(self.data_dict)-counter)/len(self.data_dict),counter/len(self.data_dict)]))**-1


    def __getitem__(self, idx):
        index_dict = self.data_dict[idx]
        extract_0,extract_1,label = index_dict[0],index_dict[1],index_dict['class']
        if not self.keep_rgb:
            extract_0,extract_1 = extract_0[:,[0,1,2,-1]],extract_1[:,[0,1,2,-1]]
        return oversample_cloud(extract_0,self.oversample)[:self.oversample,...],oversample_cloud(extract_1,self.oversample)[:self.oversample,...],torch.LongTensor([self.class_to_int(label)])