from numpy.lib.function_base import extract
import torch
import os
import numpy as np
from utils import (load_las, random_subsample,view_cloud_plotly,co_min_max,
    co_standardize,sep_standardize,extract_area,co_unit_sphere,get_voxel,get_voxel)
from torch.utils.data import Dataset, DataLoader
from itertools import permutations 
from torch_geometric.nn import fps
from tqdm import tqdm
import pandas as pd
import torch_cluster
import open3d as o3d
from .dataset_utils import registration_pipeline,context_voxel_center
class ChallengeDataset(Dataset):
    def __init__(self, csv_path,direcories_list,out_path,
    n_samples=2000,n_samples_context=2048,
    preload=False,device="cuda",context_voxel_size = [3., 3., 4.],
    final_voxel_size=[3., 3., 4.]):
        self.n_samples  = n_samples
        self.out_path = out_path
        self.voxel_size = 0.07
        self.context_voxel_size = torch.tensor(context_voxel_size)
        self.n_samples_context = n_samples_context
        self.final_voxel_size = torch.tensor(final_voxel_size)
        self.save_name = f"challenge_{self.voxel_size}.pt"
        save_path  = os.path.join(self.out_path,self.save_name)
        self.class_labels = ['nochange','removed',"added",'change',"color_change"]
        self.class_int_dict = {x:self.class_labels.index(x) for x in self.class_labels}
        self.int_class_dict = {val:key for key,val in self.class_int_dict.items()}
      
        
        df = pd.read_csv(csv_path)
        df = df[df['classification'].isin(self.class_labels)] #Remove unfit!
        scene_dicts = [{int(os.path.basename(x).split("_")[0]): os.path.join(year_path,x) for x in os.listdir(year_path) if x.split('.')[-1]=='las'} for year_path in direcories_list]
        voxel_size_icp = 0.05
        combined_scene_dicts = {x:[scene_dicts[0][x],scene_dicts[1][x]] for x in scene_dicts[0].keys()}
        if not preload :
            print(f"Recreating challenge dataset, saving to: {self.out_path}")
            
            self.pair_dict={}
            
            pair_id = 0
            self.loaded_clouds = {}
            for scene_num, scene_path_list in combined_scene_dicts.items():
                
                
                scene_list = [torch.from_numpy(load_las(scene_path_list[x])).double().to(device) for x in range(2)] 
                self.loaded_clouds[scene_num] = registration_pipeline(scene_list,voxel_size_icp,self.voxel_size)
            print(f"Saving to {save_path}!")
            torch.save(self.loaded_clouds,save_path)
        else:
            self.loaded_clouds = torch.load(save_path)
                
                
        for index, row in tqdm(df.iterrows()):
            scene_num = row['scene']
            scene_path_list=self.loaded_clouds[scene_num]
            label = self.class_int_dict[row['classification']]
            center = torch.Tensor([row['x'],row["y"]]).to(device)
            
            pair_id +=1
            self.pair_dict[pair_id] = [scene_num,center,label]
        print('Loaded dataset!')


    def __len__(self):
        return len(self.pair_dict)

    def last_processing(self, tensor_0, tensor_1):
        return co_unit_sphere(tensor_0, tensor_1)

    def get_voxels(self,cloud,context_cloud,vox_center):
        voxel_1 = get_voxel(cloud,vox_center,self.voxel_size)
        voxel_center_0 = context_voxel_center(voxel_1)
        voxel_0 = get_voxel(context_cloud,voxel_center_0,self.context_voxel_size)

        voxel_0 = voxel_0[fps(voxel_0, torch.zeros(voxel_0.shape[0]).long(
        ), ratio=self.n_samples_context/voxel_0.shape[0], random_start=False), :]
        voxel_0 = voxel_0[:self.n_samples_context,:]

        voxel_1 = voxel_1[fps(voxel_1, torch.zeros(voxel_1.shape[0]).long(
        ), ratio=self.n_samples/voxel_1.shape[0], random_start=False), :]
        voxel_1 = voxel_1[:self.n_samples,:]
        voxel_0, voxel_1 = self.last_processing(voxel_0, voxel_1)
        return voxel_0,voxel_1
    def __getitem__(self, idx):
        return_dict = {}
        scene_num,center,label = self.pair_dict[idx]
        cloud_0,cloud_1 = [extract_area(x,center,self.context_voxel_size[0].item(),
        shape='square') for x in self.loaded_clouds[scene_num]]
        voxel_height = self.voxel_size[2]
        z_max = max(cloud_0[:,2].max(),cloud_1[:,2].max())
        z_min = min(cloud_0[:,2].min(),cloud_1[:,2].min())
        z_voxel_centers = torch.arange(z_min+voxel_height/2,z_max-voxel_height/2,2*voxel_height)
        return_dict['voxels'] = {}
        for index,z_voxel_center in enumerate(z_voxel_centers.tolist()):
            vox_center = torch.cat((center,z_voxel_center),dim=-1)
            context_0, voxel_1 = self.get_voxels(cloud_1,cloud_0,vox_center)
            context_1,voxel_0 = self.get_voxels(cloud_0,cloud_1,vox_center)
            return_dict['voxels'][index] = [context_0,voxel_1,context_1,voxel_0,z_voxel_center]
        return_dict['cloud_0'] = cloud_0
        return_dict['cloud_1'] = cloud_1
        
        
        return return_dict,label