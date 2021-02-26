import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import load_las, random_subsample,view_cloud_plotly,grid_split,co_min_max
from torch.utils.data import Dataset, DataLoader
from itertools import permutations 
from torch_geometric.nn import fps
from tqdm import tqdm


eps = 1e-8



class ShapeNetLoader(Dataset):
    def __init__(self, path,out_path,sample_size=2000,preload=False,subsample='random',normalization='min_max',how_many=500):
        self.sample_size  = sample_size
        self.normalization = normalization
        self.subsample = subsample
        
        file_path_lists  = [os.path.join(path,x) for x in os.listdir(path)][0:how_many]
        self.clouds = [np.load(x) for x in file_path_lists]
     
       
            
        print('Loaded dataset!')


    def __len__(self):
        return len(self.clouds)

    def view(self,index):
        cloud = self.clouds[index]
        view_cloud_plotly(cloud[:,:3],cloud[:,3:])
        

   

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        cloud = self.clouds[idx]
        tensor_0 = random_subsample(torch.from_numpy(cloud),self.sample_size)
        tensor_1 =  random_subsample(torch.from_numpy(cloud).clone(),self.sample_size)

        if self.normalization == 'min_max':
            tensor_0[:,:3], tensor_1[:,:3] = co_min_max(tensor_0[:,:3],tensor_1[:,:3])
        elif self.normalization == 'normalize':
            concatenated = torch.cat((tensor_0[:,:3],tensor_1[:,:3]),dim=0)
            concatenated = (concatenated-concatenated.mean(axis=0))/(concatenated.std()+eps)
            tensor_0[:,:3], tensor_1[:,:3] = torch.split(concatenated,tensor_0.shape[0],dim=0)
        else:
            raise Exception('Invalid normalization type')
        
        return tensor_0,tensor_1
if __name__ == '__main__':
    dataset = ShapeNetLoader(r'D:\data\ShapeNetCore.v2.PC15k\02691156\train',sample_size=2000,preload=True,subsample='random',normalization='min_max')
    print()