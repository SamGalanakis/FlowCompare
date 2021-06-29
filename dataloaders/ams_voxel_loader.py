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
                   rotate_xy,get_voxel,get_all_voxel_centers,get_voxel_center)

from itertools import combinations
from torch_cluster import fps
from tqdm import tqdm
from torch.utils.data import Dataset
import json
from datetime import datetime
import random
import open3d as o3d
import torch_cluster

eps = 1e-8





def filter_scans(scans_list, dist):
    print(f"Filtering scans")
    ignore_list = []
    keep_scans = []
    for scan in tqdm(scans_list):
        if scan in ignore_list:
            continue
        else:
            keep_scans.append(scan)
        ignore_list.extend(
            [x for x in scans_list if np.linalg.norm(x.center-scan.center) < dist])
    return keep_scans


class Scan:
    def __init__(self, recording_properties, base_dir):
        self.recording_properties = recording_properties
        self.id = self.recording_properties['ImageId']
        self.center = np.array(
            [self.recording_properties['X'], self.recording_properties['Y']])
        self.height = self.recording_properties['Height']
        self.ground_offset = self.recording_properties['GroundLevelOffset']
        self.ground_height = self.height-self.ground_offset
        self.path = os.path.join(base_dir, f'{self.id}.laz')
        self.datetime = datetime(int(self.recording_properties['RecordingTimeGps'].split('-')[0]), int(
            self.recording_properties['RecordingTimeGps'].split('-')[1]), int(self.recording_properties['RecordingTimeGps'].split('-')[-1].split('T')[0]))


class AmsVoxelLoader(Dataset):
    def __init__(self, directory_path_train,directory_path_test, out_path, clearance=10, preload=False,
                 height_min_dif=0.5, max_height=15.0, device="cpu", ground_keep_perc=1/40, n_samples=2048, n_voxels=10, final_voxel_size=[3., 3., 4.],
                 rotation_augment = True,n_samples_context=2048, context_voxel_size = [3., 3., 4.],
                mode='train',verbose=False,voxel_size_final_downsample=0.07,getter_mode='sample'):

        print(f'Dataset mode: {mode}, getter_mode : {getter_mode}')
        self.mode = mode
        self.getter_mode = getter_mode
        if self.mode =='train':
            directory_path = directory_path_train
        elif self.mode == 'test':
            directory_path = directory_path_test
        else:
            raise Exception('Invalid mode')
        self.verbose = verbose 
        self.voxel_size_final_downsample = voxel_size_final_downsample
        self.n_samples_context = n_samples_context
        self.context_voxel_size = torch.tensor(context_voxel_size)
        self.directory_path = directory_path
        self.clearance = clearance
        
        self.out_path = out_path
        self.height_min_dif = height_min_dif
        self.max_height = max_height
        self.rotation_augment = rotation_augment
        self.save_name = f'ams_{mode}_save_dict_{clearance}.pt'
        name_insert = self.save_name.split('.')[0]
        self.filtered_scan_path = os.path.join  (
            out_path, f'{name_insert}_filtered_scans.pt')
        self.all_valid_combs_path = os.path.join  (
            out_path, f'{name_insert}_all_valid_combs.pt')
        self.years = [2019, 2020]
        self.ground_keep_perc = ground_keep_perc
        self.over_ground_cutoff = 0.1
        
        self.n_samples = n_samples
        self.final_voxel_size = torch.tensor(final_voxel_size)
        self.n_voxels = n_voxels
        save_path = os.path.join(self.out_path, self.save_name)
        voxel_size_icp = 0.05


        random.seed(0)

        if not preload:

            with open(os.path.join(directory_path, 'args.json')) as f:
                self.args = json.load(f)
            with open(os.path.join(directory_path, 'response.json')) as f:
                self.response = json.load(f)

            print(f"Recreating dataset, saving to: {self.out_path}")
            self.scans = [Scan(x, self.directory_path)
                          for x in self.response['RecordingProperties']]
            self.scans = [
                x for x in self.scans if x.datetime.year in self.years]
            if os.path.isfile(self.filtered_scan_path):
                with open(self.filtered_scan_path, "rb") as fp:
                    self.filtered_scans = pickle.load(fp)
            else:
                self.filtered_scans = filter_scans(self.scans, 3)
                with open(self.filtered_scan_path, "wb") as fp:
                    pickle.dump(self.filtered_scans, fp)

            self.save_dict = {}
            save_id = -1

            for scene_number, scan in enumerate(tqdm(self.filtered_scans)):
                # Gather scans within certain distance of scan center
                relevant_scans = [x for x in self.scans if np.linalg.norm(
                    x.center-scan.center) < 7]

                relevant_times = set([x.datetime for x in relevant_scans])

                # Group by dates
                time_partitions = {time: [
                    x for x in relevant_scans if x.datetime == time] for time in relevant_times}

                # Load and combine clouds from same date
                clouds_per_time = [torch.from_numpy(np.concatenate([load_las(
                    x.path) for x in val])).double().to(device) for key, val in time_partitions.items()]

                # Make xy 0 at center to avoid large values
                center_trans = torch.cat(
                    (torch.from_numpy(scan.center), torch.tensor([0, 0, 0, 0]))).double().to(device)
                
                clouds_per_time = [x-center_trans for x in clouds_per_time]
                # Extract square at center since only those will be used for grid
                clouds_per_time = [x[extract_area(x, center=np.array(
                    [0, 0]), clearance=self.clearance, shape='square'), :] for x in clouds_per_time]
                # Apply registration between each cloud and first in list, store transforms
                # First cloud does not need to be transformed
                

                clouds_per_time = registration_pipeline(
                    clouds_per_time, voxel_size_icp, self.voxel_size_final_downsample)

               
                # Remove below ground and above cutoff
                # Cut off slightly under ground height
        

                ground_cutoff = scan.ground_height - 0.05 
                height_cutoff = ground_cutoff+max_height 
                clouds_per_time = [x[torch.logical_and(
                    x[:, 2] > ground_cutoff, x[:, 2] < height_cutoff), ...] for x in clouds_per_time]

                clouds_per_time = [x.float().cpu() for x in clouds_per_time]

                save_id += 1
                save_entry = {'clouds': clouds_per_time,
                              'ground_height': scan.ground_height}

                save_id += 1
                save_entry = {'clouds': clouds_per_time,
                              'ground_height': scan.ground_height}

                self.save_dict[save_id] = save_entry
                if scene_number % 100 == 0 and scene_number != 0:
                    print(f"Progressbackup: {scene_number}!")
                    torch.save(self.save_dict, save_path)

            print(f"Saving to {save_path}!")
            torch.save(self.save_dict, save_path)
        else:
            self.save_dict = torch.load(save_path)


        if self.getter_mode == 'all':
            
            if os.path.isfile(self.all_valid_combs_path):
                with open(self.all_valid_combs_path, "rb") as fp:
                    self.all_valid_combs = pickle.load(fp)
            else:
                self.all_valid_combs = []
                for idx, (save_id,save_entry) in enumerate(tqdm(self.save_dict.items())):
                    clouds = save_entry['clouds']
                    ground_height = save_entry['ground_height']
           
                    clouds = [x for x in clouds if x.shape[0] > 5000]
                    if len(clouds) < 2:
                        if self.verbose:
                            print(f'Not enough clouds {idx}, skipping ')
                        continue
                    cluster_min = clouds[0].min(axis=0)[0][:3]
                    cluster_max = clouds[0].max(axis=0)[0][:3]
                    clusters = [torch_cluster.grid_cluster(
                        x[:, :3],start= cluster_min,end=cluster_max,size= self.final_voxel_size) for x in clouds]


                    
                
                    valid_voxels = []
                    for cluster in clusters:
                        cluster_indices, counts = cluster.unique(return_counts=True)
                        valid_indices = cluster_indices[counts > self.n_samples_context]
                        valid_voxels.append(valid_indices)
                    common_voxels = []
                    for ind_0, ind_1 in combinations(range(0, len(clouds)), 2):
                        if ind_0 == ind_1:
                            continue
                        common_voxels.append([ind_0, ind_1, np.intersect1d(
                            valid_voxels[ind_0], valid_voxels[ind_1]).tolist()])
                    valid_combs = []
                    for val in common_voxels:
                        valid_combs.extend([(val[0], val[1], x) for x in val[2]])
                        # Self predict (only on index,since clouds shuffled and 1:1 other to same)
                        valid_combs.extend([(val[0], val[0], x) for x in val[2]])

                    if len(valid_combs) < 1:
                        # If not enough recursively give other index from dataset
                        if self.verbose:
                            print(f"Couldn't find combinations for index: {idx}")
                        continue
                    

                    
                    found_valid = False
                    for comb_ind,combination in enumerate(valid_combs):
                        
                        cloud_ind_1 = combination[1]
                        indices = clusters[cloud_ind_1] == combination[2]
                        
                        voxel_1 = clouds[cloud_ind_1][indices, :]
                        voxel_center = get_voxel_center(voxel_1[0,:3],cluster_min,self.final_voxel_size)
                        cloud_ind_0 = combination[0]
                        voxel_0 = get_voxel(clouds[cloud_ind_0],voxel_center,self.context_voxel_size)
                        if voxel_0.shape[0]>=self.n_samples_context:
                            self.all_valid_combs.append({'combination':(save_id,)+combination,'cluster_min':cluster_min,'cluster_max':cluster_max})
                        
                    
                    if not found_valid:
                        if self.verbose:
                            print(f"Couldn't find valid context for any tile: {idx}, skipping")
                        continue
                with open(self.all_valid_combs_path, "wb") as fp:
                    pickle.dump(self.all_valid_combs, fp)
                        
                    

        print('Loaded dataset!')

    def __len__(self):
        if self.getter_mode =='sample':
            return len(self.save_dict)
        elif self.getter_mode == 'all':
            return len(self.all_valid_combs)

    def sample_voxel_pairs(self,idx):
        clouds = self.save_dict[idx]['clouds']
        ground_height = self.save_dict[idx]['ground_height']
        random.shuffle(clouds)
        
        clouds = [x for x in clouds if x.shape[0] > 5000]
        if len(clouds) < 2:
            if self.verbose:
                print(f'Not enough clouds {idx}, recursive return ')
            return self.__getitem__(random.randint(0, self.__len__()-1))
        cluster_min = clouds[0].min(axis=0)[0][:3]
        cluster_max = clouds[0].max(axis=0)[0][:3]
        clusters = [torch_cluster.grid_cluster(
            x[:, :3],start= cluster_min,end=cluster_max,size= self.final_voxel_size) for x in clouds]

        valid_voxels = []
        for cluster in clusters:
            cluster_indices, counts = cluster.unique(return_counts=True)
            valid_indices = cluster_indices[counts > self.n_samples_context]
            valid_voxels.append(valid_indices)
        common_voxels = []
        for ind_0, ind_1 in combinations(range(0, len(clouds)), 2):
            if ind_0 == ind_1:
                continue
            common_voxels.append([ind_0, ind_1, np.intersect1d(
                valid_voxels[ind_0], valid_voxels[ind_1]).tolist()])
        valid_combs = []
        for val in common_voxels:
            valid_combs.extend([(val[0], val[1], x) for x in val[2]])
            # Self predict (only on index,since clouds shuffled and 1:1 other to same)
            valid_combs.extend([(val[0], val[0], x) for x in val[2]])

        if len(valid_combs) < 1:
            # If not enough recursively give other index from dataset
            if self.verbose:
                print(f"Couldn't find combinations for index: {idx}")
            return self.__getitem__(random.randint(0, self.__len__()-1))
        

        random.shuffle(valid_combs)
        found_valid = False
        for draw_ind,draw in enumerate(valid_combs):
            
            cloud_ind_1 = draw[1]
            indices = clusters[cloud_ind_1] == draw[2]
            
            voxel_1 = clouds[cloud_ind_1][indices, :]
            voxel_center = get_voxel_center(voxel_1[0,:3],cluster_min,self.final_voxel_size)
            cloud_ind_0 = draw[0]
            voxel_0 = get_voxel(clouds[cloud_ind_0],voxel_center,self.context_voxel_size)
            if not voxel_0.shape[0]>=self.n_samples_context:
                continue
            else:
                found_valid = True
                break
        
        if not found_valid:
            if self.verbose:
                print(f"Couldn't find valid context for any tile: {idx}")
            return self.__getitem__(random.randint(0, self.__len__()-1))
            
        voxel_1 = voxel_1[fps(voxel_1, torch.zeros(voxel_1.shape[0]).long(
        ), ratio=self.n_samples/voxel_1.shape[0], random_start=False), :]
        voxel_1 = voxel_1[:self.n_samples, :]

        

        
        are_same = (cloud_ind_1 == cloud_ind_0)
        
     
        voxel_0 = voxel_0[fps(voxel_0, torch.zeros(voxel_0.shape[0]).long(
        ), ratio=self.n_samples_context/voxel_0.shape[0], random_start=False), :]
        voxel_0 = voxel_0[:self.n_samples_context,:]
        
        if are_same:
            # Add rgb noise 0-1 cm when same cloud
            voxel_1 = voxel_1.clone()
            voxel_0[:, :3] += torch.rand_like(voxel_0[:, :3])*0.01

        tensor_0, tensor_1,inverse  = self.last_processing(voxel_0, voxel_1)
        rads = torch.rand((1))*math.pi*2

        if self.rotation_augment:
            rot_mat = rotate_xy(rads)
            tensor_0[:, :2] = torch.matmul(tensor_0[:, :2], rot_mat)
            tensor_1[:, :2] = torch.matmul(tensor_1[:, :2], rot_mat)
        # Distance from ground as extra context
        extra_context = inverse['mean'][2] - ground_height
        extra_context = extra_context.unsqueeze(-1)

        return tensor_0, tensor_1,extra_context

    def all_getter(self,idx):



        combination = self.all_valid_combs[idx]
        save_id,cloud_ind_0,cloud_ind_1,common_voxel = combination
        clouds = self.save_dict[save_id]['clouds']
        ground_height = self.save_dict[save_id]['ground_height']

        clouds = [clouds[cloud_ind_0],clouds[cloud_ind_1]]

        cluster_min = clouds[0].min(axis=0)[0][:3]
        cluster_max = clouds[0].max(axis=0)[0][:3]
        clusters = [torch_cluster.grid_cluster(
            x[:, :3],start= cluster_min,end=cluster_max,size= self.final_voxel_size) for x in clouds]
        
        
        indices = clusters[1] == common_voxel
        
        voxel_1 = clouds[1][indices, :]
        voxel_center = get_voxel_center(voxel_1[0,:3],cluster_min,self.final_voxel_size)
        
        voxel_0 = get_voxel(clouds[0],voxel_center,self.context_voxel_size)

        voxel_1 = voxel_1[fps(voxel_1, torch.zeros(voxel_1.shape[0]).long(
        ), ratio=self.n_samples/voxel_1.shape[0], random_start=False), :]
        voxel_1 = voxel_1[:self.n_samples, :]

        are_same = (cloud_ind_1 == cloud_ind_0)
        
    
        voxel_0 = voxel_0[fps(voxel_0, torch.zeros(voxel_0.shape[0]).long(
        ), ratio=self.n_samples_context/voxel_0.shape[0], random_start=False), :]
        voxel_0 = voxel_0[:self.n_samples_context,:]
        #Only augment in train
        if are_same:
            voxel_1 = voxel_1.clone()
            if self.mode == 'train':
                voxel_0[:, :3] += torch.rand_like(voxel_0[:, :3])*0.01
        

        tensor_0, tensor_1,inverse  = self.last_processing(voxel_0, voxel_1)

        if self.mode == 'train':
            rads = torch.rand((1))*math.pi*2

            if self.rotation_augment:
                rot_mat = rotate_xy(rads)
                tensor_0[:, :2] = torch.matmul(tensor_0[:, :2], rot_mat)
                tensor_1[:, :2] = torch.matmul(tensor_1[:, :2], rot_mat)
        
        # Distance from ground as extra context
        extra_context = inverse['mean'][2] - ground_height
        extra_context = extra_context.unsqueeze(-1)

        return tensor_0, tensor_1,extra_context


    def last_processing(self, tensor_0, tensor_1):
        return co_unit_sphere(tensor_0, tensor_1,return_inverse=True)

    def __getitem__(self, idx):

        if self.getter_mode == 'sample':
            return self.sample_voxel_pairs(idx)
        elif self.getter_mode == 'all':
            return self.all_getter(idx)
        
        
