import torch
import matplotlib.pyplot as plt
import os
import math
import numpy as np
import pykeops
import pickle
from .dataset_utils import registration_pipeline, context_voxel_center
from utils import (load_las,
                   co_unit_sphere,
                   extract_area,
                   rotate_xy,get_voxel,get_all_voxel_centers,get_voxel_center, view_cloud_plotly)

from itertools import combinations, combinations_with_replacement,product
from torch_cluster import fps
from tqdm import tqdm
from torch.utils.data import Dataset
import json
from datetime import datetime
import random
import open3d as o3d
import torch_cluster
from utils import voxelize
from .ams_voxel_loader import filter_scans,Scan
eps = 1e-8





class FullSceneLoader(Dataset):
    def __init__(self, directory_path_train,directory_path_test, out_path, clearance=10, 
                  device="cpu", n_samples=1024,final_voxel_size=[3., 3., 4.],
                 n_samples_context=1250, context_voxel_size = [3., 3., 4.],
                mode='train'):

        print(f'Dataset mode: {mode}')
        self.mode = mode
     
        if self.mode =='train':
            directory_path = directory_path_train
        elif self.mode == 'test':
            directory_path = directory_path_test
        else:
            raise Exception('Invalid mode')
   
        self.n_samples_context = n_samples_context
        self.context_voxel_size = torch.tensor(context_voxel_size)
        self.directory_path = directory_path
        self.clearance = clearance
        
        self.out_path = out_path
        self.save_name = f'ams_{mode}_save_dict_{clearance}.pt'
        name_insert = self.save_name.split('.')[0]
        self.filtered_scan_path = os.path.join  (
            out_path, f'{name_insert}_filtered_scans.pt')
        self.all_valid_combs_path = os.path.join  (
            out_path, f'{name_insert}_all_valid_combs.pt')
        self.years = [2019, 2020]

        
        self.n_samples = n_samples
        self.final_voxel_size = torch.tensor(final_voxel_size)
        save_path = os.path.join(self.out_path, self.save_name)
       

        self.save_dict = torch.load(save_path)


        self.save_dict = {i:val for i , (key,val) in enumerate(self.save_dict.items())}
            
        

                    

        print('Loaded dataset!')

    def __len__(self):
        return len(self.save_dict)
      

    
    def view(self,idx):
        tensor_0, tensor_1,extra_context = self.all_getter(idx)
        fig_0 = view_cloud_plotly(tensor_0[:,:3],tensor_0[:,3:],point_size=10,show=False)
        fig_1 = view_cloud_plotly(tensor_1[:,:3],tensor_1[:,3:],point_size=10,show=False)
        
        return fig_0,fig_1
    def get_scene(self,idx):
        file_path = f'save/processed_dataset/preprocessed_scenes/processed_scene_{idx}.pt'
        if os.path.exists(file_path):
            out_list = torch.load(file_path)
            
        else:
            save_entry = self.save_dict[idx]
            clouds = save_entry['clouds']
            ground_height = save_entry['ground_height']


            clouds = clouds[:2]
            cluster_min = torch.stack([torch.min(x,dim=0)[0][:3] for x in clouds]).min(dim=0)[0]
            cluster_max = torch.stack([torch.max(x,dim=0)[0][:3] for x in clouds]).max(dim=0)[0]
            cloud_0,cloud_1 = clouds
            clusters = []
            
            for index,x in enumerate(clouds):
                    labels,voxel_centers = voxelize(x[:, :3],start= cluster_min,end=cluster_max,size= self.final_voxel_size)
                    clusters.append(labels)
            voxel_indexes = torch.cat(clusters).unique()
            voxel_0_list = []
            voxel_0_context_list = []
            voxel_1_list = []
            voxel_1_context_list = []
            extra_context_list = []
            for voxel_index in tqdm(voxel_indexes.tolist()):
                center = voxel_centers[index]
                voxel_1_context = get_voxel(cloud_1,center,self.context_voxel_size)
                voxel_0_context = get_voxel(cloud_0,center,self.context_voxel_size)
            
                voxel_1 = cloud_1[(clusters[1]==voxel_index).squeeze(),:]
                voxel_0 = cloud_0[(clusters[0]==voxel_index).squeeze(),:]

                if voxel_0.shape[0]<2:
                    voxel_1_mean = voxel_1.mean(dim=0)
                    voxel_0 = torch.stack([voxel_1_mean,voxel_1_mean],dim=0)

                if voxel_1.shape[0]<2:
                    voxel_0_mean = voxel_0.mean(dim=0)
                    voxel_1 = torch.stack([voxel_0_mean,voxel_0_mean],dim=0)

                if voxel_0_context.shape[0]<2:
                    voxel_0_context = voxel_0

                if voxel_1_context.shape[0]<2:
                    voxel_1_context = voxel_1

                


                voxel_0_context = voxel_0_context[fps(voxel_0_context, torch.zeros(voxel_0_context.shape[0]).long(
                ), ratio=self.n_samples_context/voxel_0_context.shape[0], random_start=False), :]
                voxel_0_context = voxel_0_context[:self.n_samples_context, :]


                voxel_1_context = voxel_1_context[fps(voxel_1_context, torch.zeros(voxel_1_context.shape[0]).long(
                ), ratio=self.n_samples_context/voxel_1_context.shape[0], random_start=False), :]
                voxel_1_context = voxel_1_context[:self.n_samples_context, :]


                voxel_1 = voxel_1[fps(voxel_1, torch.zeros(voxel_1.shape[0]).long(
                ), ratio=self.n_samples/voxel_1.shape[0], random_start=False), :]
                voxel_1 = voxel_1[:self.n_samples, :]

                
                
            
                voxel_0 = voxel_0[fps(voxel_0, torch.zeros(voxel_0.shape[0]).long(
                ), ratio=self.n_samples_context/voxel_0.shape[0], random_start=False), :]
                voxel_0 = voxel_0[:self.n_samples_context,:]
            
                

                voxel_0_context_list.append(voxel_0_context)
                voxel_1_context_list.append(voxel_1_context)
                voxel_0_list.append(voxel_0)
                voxel_1_list.append(voxel_1)

                
                # Distance from ground as extra context
                extra_context = center[2] - ground_height
                extra_context = extra_context.unsqueeze(-1)
                extra_context_list.append(extra_context)
                out_list = [voxel_0_list, voxel_1_list,voxel_0_context_list,voxel_1_context_list,extra_context_list]
                torch.save(out_list,file_path)
        voxel_0_list, voxel_1_list,voxel_0_context_list,voxel_1_context_list,extra_context_list = out_list
        return voxel_0_list, voxel_1_list,voxel_0_context_list,voxel_1_context_list,extra_context_list


    def last_processing(self, tensor_0, tensor_1):
        return co_unit_sphere(tensor_0, tensor_1,return_inverse=True)

    def __getitem__(self, idx):

        return self.get_scene(idx)
   
        
        
