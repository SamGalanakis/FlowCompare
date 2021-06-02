import torch
import matplotlib.pyplot as plt
import os
import math
import numpy as np
import pykeops
import pickle
from utils import (load_las, 
random_subsample,
view_cloud_plotly,
grid_split,co_min_max,
circle_split,co_standardize,
sep_standardize,
co_unit_sphere,
extract_area,
rotate_xy)
from itertools import permutations 
from torch_geometric.nn import fps
from tqdm import tqdm
from torch.utils.data import Dataset
import json
from datetime import datetime
import random
import cupoch as cph

eps = 1e-8




def is_only_ground(cloud,perc=0.99):
    clouz_z = cloud[:,2]
    cloud_min = clouz_z.min()
    n_close_ground = (clouz_z<(cloud_min + 0.3)).sum()
    
    return n_close_ground/cloud.shape[0] >=perc


def voxel_downsample(cloud,voxel_size):
    pcd = cph.geometry.PointCloud()
    pcd.points = cph.utility.Vector3fVector(cloud.cpu().numpy()[:,:3])
    pcd.colors = cph.utility.Vector3fVector(cloud.cpu().numpy()[:,3:])
    pcd = pcd.voxel_down_sample(voxel_size)
    cloud_ = np.concatenate((np.asarray(pcd.points.cpu()),np.asarray(pcd.colors.cpu())),axis=-1)
    return torch.from_numpy(cloud_).to(cloud.device)



def filter_scans(scans_list,dist):
    print(f"Filtering scans")
    ignore_list = []
    keep_scans=[]
    for scan in tqdm(scans_list):
        if scan in ignore_list:
            continue
        else: 
            keep_scans.append(scan)
        ignore_list.extend([x for x in scans_list if np.linalg.norm(x.center-scan.center)<dist])
    return keep_scans

class Scan:
    def __init__(self,recording_properties,base_dir):
        self.recording_properties = recording_properties
        self.id = self.recording_properties['ImageId']
        self.center = np.array([self.recording_properties['X'],self.recording_properties['Y']])
        self.height = self.recording_properties['Height']
        self.ground_offset = self.recording_properties['GroundLevelOffset']
        self.ground_height = self.height-self.ground_offset
        self.path = os.path.join(base_dir,f'{self.id}.laz')
        self.datetime = datetime(int(self.recording_properties['RecordingTimeGps'].split('-')[0]),int(self.recording_properties['RecordingTimeGps'].split('-')[1]),int(self.recording_properties['RecordingTimeGps'].split('-')[-1].split('T')[0]))

class AmsGridLoaderPointwise(Dataset):
    def __init__(self, directory_path,out_path,grid_square_size = 2,clearance = 10,preload=False,min_points=500,
    height_min_dif=0.5,max_height = 15.0, device="cuda",ground_perc=0.90,ground_keep_perc=1/40,voxel_size=0.07):
        self.directory_path = directory_path
        self.grid_square_size = grid_square_size
        self.clearance = clearance
        self.filtered_scan_path = os.path.join(out_path,'filtered_scan_path.pt')
        self.min_points = min_points
        self.out_path = out_path
        self.height_min_dif = height_min_dif
        self.max_height = max_height
        self.minimum_difs = torch.Tensor([self.grid_square_size*0.9,self.grid_square_size*0.9,self.height_min_dif]).to(device)
        self.save_name = f"pointwise_ams_extract_id_dict_{clearance}_{self.min_points}_{self.grid_square_size}_{voxel_size}_{self.height_min_dif}.pt"
        self.filtered_path = f'pointwise_filtered_{ground_perc}_{ground_keep_perc}_'+self.save_name
        self.years = [2019,2020]
        self.ground_perc = ground_perc
        self.ground_keep_perc = ground_keep_perc
        self.voxel_size = voxel_size

        
        
        save_path  = os.path.join(self.out_path,self.save_name)
        if not preload:

            with open(os.path.join(directory_path,'args.json')) as f:
                self.args = json.load(f)
            with open(os.path.join(directory_path,'response.json')) as f:
                self.response = json.load(f)


            print(f"Recreating dataset, saving to: {self.out_path}")
            self.scans = [Scan(x,self.directory_path) for x in self.response['RecordingProperties']]
            self.scans = [x for x in self.scans if x.datetime.year in self.years]
            if os.path.isfile(self.filtered_scan_path):
                with open(self.filtered_scan_path, "rb") as fp:
                    self.filtered_scans = pickle.load(fp)
            else:   
                self.filtered_scans = filter_scans(self.scans,3)
                with open(self.filtered_scan_path, "wb") as fp:  
                    pickle.dump(self.filtered_scans, fp)
            

            
            
            self.save_dict={}
            save_id = -1
            
            for scene_number, scan in enumerate(tqdm(self.filtered_scans)):
                relevant_scans = [x for x in self.scans if np.linalg.norm(x.center-scan.center)<7]
                relevant_times = set([x.datetime for x in relevant_scans])
                
                time_partitions = {time:[x for x in relevant_scans if x.datetime ==time] for time in relevant_times}
                
                
                clouds_per_time = [torch.from_numpy(np.concatenate([load_las(x.path) for x in val])).float().to(device) for key,val in time_partitions.items()]

                ground_cutoff = scan.ground_height - 0.05 # Cut off slightly under ground height
                height_cutoff = ground_cutoff+max_height
                clouds_per_time = [x[torch.logical_and(x[:,2]>ground_cutoff,x[:,2]<height_cutoff),...] for x in clouds_per_time]

                clouds_per_time = [voxel_downsample(x,self.voxel_size) for x in clouds_per_time]
     
                grids = [grid_split(cloud,self.grid_square_size,center=scan.center,clearance = self.clearance) for cloud in clouds_per_time]
                
                grid_masks = [[self.valid_tile(tile) for tile in grid_list] for grid_list in grids]
                

                
                valid_grid_masks =[]
                valid_grids = []

                for grid_mask,grid in zip(grid_masks,grids):
                    if sum(grid_mask)>20:
                        valid_grid_masks.append(grid_mask)
                        grid = [x.cpu() for x in grid] #Put on gpu before next
                        valid_grids.append(grid)
                
                if len(valid_grids)<2:
                    print(f"Skipping scene")
                    continue
                
                save_id+=1
                save_entry = {'grids':valid_grids,'grid_masks':valid_grid_masks,'ground_height':scan.ground_height}

                self.save_dict[save_id] = save_entry
            


                        
            
            print(f"Saving to {save_path}!")
            torch.save(self.save_dict,save_path)
        else:
            self.save_dict = torch.load(save_path)


        #Combinations of form # (scene_index,grid_index,grid_index)
        self.combinations_list = []
        for entry_index,entry in enumerate(self.save_dict.items()):
            index_permutations = list(permutations(range(len(entry['grids'])),2))
            for perm in index_permutations:
                unique_combination = list(perm)
                unique_combination.insert(0,entry_index)
                self.combinations_list.append(unique_combination)
            for x in range(len(entry['grids'])):
                self.combinations_list.append([entry_index,x,x])
        print('Loaded dataset!')

    def valid_tile(self,tile):
        min_points_bool = tile.shape[0]>=self.min_points
        if not min_points_bool:
            return False
        coverage_bool = ((tile.max(dim=0)[0][:3]-tile.min(dim=0)[0][:3] )>self.minimum_difs).all().item()
        return min_points_bool and coverage_bool

    def __len__(self):
        return len(self.combinations_list)


    def last_processing(self,tensor_0,tensor_1,normalization):
        
        
        if normalization == 'min_max':
            tensor_0[:,:3], tensor_1[:,:3] = co_min_max(tensor_0[:,:3],tensor_1[:,:3])
        elif normalization == 'co_unit_sphere':
            tensor_0,tensor_1 = co_unit_sphere(tensor_0,tensor_1)
        elif normalization == 'standardize':
            tensor_0,tensor_1 = co_standardize(tensor_0,tensor_1)
        elif normalization == 'sep_standardize':
            tensor_0,tensor_1 = sep_standardize(tensor_0,tensor_1)
        else:
            raise Exception('Invalid normalization type')
        return tensor_0,tensor_1
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        combination_entry = self.combinations_list[idx]
        relevant_tensors = self.extract_id_dict[combination_entry[0]]
        #CLONE THE TENSOR IF SAME, OTHERWISE POINT TO SAME MEMORY, PROBLEMS IN NORMALIZATION
        if combination_entry[1]!=combination_entry[2]:
            tensor_0 = relevant_tensors[combination_entry[1]]
            tensor_1 = relevant_tensors[combination_entry[2]]
        else:
            tensor_0 = relevant_tensors[combination_entry[1]]
            tensor_1 = relevant_tensors[combination_entry[2]].clone()
            if self.augment_same:
                tensor_0[:,:3] += torch.rand_like(tensor_0[:,:3])*0.01 #Add rgb noise 0-1 cm when same cloud
        #Remove pesky extra points due to fps ratio
        tensor_0 = tensor_0[:self.min_points,:]
        tensor_1 = tensor_1[:self.min_points,:]
        tensor_0,tensor_1 = self.last_processing(tensor_0,tensor_1,self.normalization)
        rads  = torch.rand((1))*math.pi*2

        if self.rotation_augment:
            rot_mat = rotate_xy(rads)
            tensor_0[:,:2] = torch.matmul(tensor_0[:,:2],rot_mat)
            tensor_1[:,:2] = torch.matmul(tensor_1[:,:2],rot_mat)
        
        return tensor_0,tensor_1





