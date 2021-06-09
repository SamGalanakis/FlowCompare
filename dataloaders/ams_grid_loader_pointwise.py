from o3d_reg import draw_registration_result
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
rotate_xy,
view_cloud_o3d)
from itertools import permutations 
from torch_cluster import fps
from tqdm import tqdm
from torch.utils.data import Dataset
import json
from datetime import datetime
import random
import cupoch as cph
import open3d as o3d
import torch_cluster
eps = 1e-8




def is_only_ground(cloud,perc=0.99):
    clouz_z = cloud[:,2]
    cloud_min = clouz_z.min()
    n_close_ground = (clouz_z<(cloud_min + 0.3)).sum()
    
    return n_close_ground/cloud.shape[0] >=perc

def icp_reg_precomputed_target(source_cloud,target,voxel_size=0.05,max_it=2000):
    source_cloud = source_cloud.cpu().numpy()
    threshold = voxel_size * 0.4
    trans_init = np.eye(4).astype(np.float32)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_cloud[:,:3])
    source.colors = o3d.utility.Vector3dVector(source_cloud[:,3:])
    source = source.voxel_down_sample(voxel_size)
    source.estimate_normals()
    result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_it))

    return result

def downsample_transform(cloud,voxel_size,transform):
    device = cloud.device
    source = o3d.geometry.PointCloud()
    cloud = cloud.cpu().numpy()
    source.points = o3d.utility.Vector3dVector(cloud[:,:3])
    source.colors = o3d.utility.Vector3dVector(cloud[:,3:])
    source = source.voxel_down_sample(voxel_size)
    source.transform(transform)
    cloud = torch.from_numpy(np.concatenate((np.asarray(source.points),np.asarray(source.colors)),axis=-1))
    return cloud.to(device)
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
    def __init__(self, directory_path,out_path,clearance = 10,preload=False,
    height_min_dif=0.5,max_height = 15.0, device="cpu",ground_keep_perc=1/40,voxel_size=0.07,n_samples=2048,n_voxels=10,final_voxel_size=[3.,3.,4.]):
        self.directory_path = directory_path
        self.clearance = clearance
        self.filtered_scan_path = os.path.join(out_path,'filtered_scan_path.pt')
        self.out_path = out_path
        self.height_min_dif = height_min_dif
        self.max_height = max_height
        
        self.save_name = f"pointwise_ams_save_dict_{clearance}.pt"
        self.years = [2019,2020]
        self.ground_keep_perc = ground_keep_perc
        self.over_ground_cutoff = 0.1 
        self.voxel_size = voxel_size
        self.n_samples = n_samples
        self.final_voxel_size = torch.tensor(final_voxel_size)
        self.n_voxels = n_voxels
        
        
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
                #Gather scans within certain distance of scan center
                relevant_scans = [x for x in self.scans if np.linalg.norm(x.center-scan.center)<7]
                
                relevant_times = set([x.datetime for x in relevant_scans])

                #Group by dates
                time_partitions = {time:[x for x in relevant_scans if x.datetime ==time] for time in relevant_times}
                
                #Load and combine clouds from same date
                clouds_per_time = [torch.from_numpy(np.concatenate([load_las(x.path) for x in val])).double().to(device) for key,val in time_partitions.items()]
                

                #Make xy 0 at center to avoid large values
                center_trans = torch.cat((torch.from_numpy(scan.center),torch.tensor([0,0,0,0]))).double().to(device)
                
                clouds_per_time = [x-center_trans for x in clouds_per_time]
                # Extract square at center since only those will be used for grid
                clouds_per_time = [x[extract_area(x,center = np.array([0,0]),clearance = self.clearance,shape='square'),:] for x in clouds_per_time]
                #Apply registration between each cloud and first in list, store transforms
                registration_transforms = [np.eye(4,dtype=np.float32)] #First cloud does not need to be transformed


                voxel_size_icp = 0.05
                target_cloud = clouds_per_time[0].cpu().numpy()
                target = o3d.geometry.PointCloud()
                target.points = o3d.utility.Vector3dVector(target_cloud[:,:3])
                target.colors = o3d.utility.Vector3dVector(target_cloud[:,3:])
                target = target.voxel_down_sample(voxel_size_icp)
                target.estimate_normals()
                for source_cloud in clouds_per_time[1:]:
                    result=icp_reg_precomputed_target(source_cloud,target,voxel_size=voxel_size_icp)
                    registration_transforms.append(result.transformation)
                
                

                #Remove below ground and above cutoff 
                ground_cutoff = scan.ground_height - 0.05 # Cut off slightly under ground height
                height_cutoff = ground_cutoff+max_height
                clouds_per_time = [x[torch.logical_and(x[:,2]>ground_cutoff,x[:,2]<height_cutoff),...] for x in clouds_per_time]
                #Downsample and apply registration
                clouds_per_time = [downsample_transform(x,self.voxel_size,transform) for x,transform in zip(clouds_per_time,registration_transforms)]
                clouds_per_time = [x.float().cpu() for x in clouds_per_time]

              
                
                save_id+=1
                save_entry = {'clouds':clouds_per_time,'ground_height':scan.ground_height}

                self.save_dict[save_id] = save_entry
                if scene_number % 100 == 0 and scene_number!= 0 :
                    print(f"Progressbackup: {scene_number}!")
                    torch.save(self.save_dict,save_path)
            


                        
            
            print(f"Saving to {save_path}!")
            torch.save(self.save_dict,save_path)
        else:
            self.save_dict = torch.load(save_path)


    
        print('Loaded dataset!')


    def __len__(self):
        return len(self.save_dict)
    def view(self,index,grid_ind_0=0,grid_ind_1=1):
        clouds = self.save_dict[index]['clouds']
        cloud_0,cloud_1 = clouds[grid_ind_0],clouds[grid_ind_1]
        
        a_,b_ = o3d.geometry.PointCloud(),o3d.geometry.PointCloud()
        a_.points = o3d.utility.Vector3dVector(cloud_0[:,:3])
        b_.points = o3d.utility.Vector3dVector(cloud_1[:,:3])
        draw_registration_result(a_,b_,np.eye(4))

    
    def last_processing(self,samples,context):

        samples,context = co_unit_sphere(samples,context)
        return samples,context
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        
        clouds = self.save_dict[idx]['clouds']
        
        clouds = [x for x in clouds if x.shape[0]>5000]
        if len(clouds)<2:
            print(f'Not enough clouds {idx}, recursive return ')
            return self.__getitem__(random.randint(0,self.__len__()-1))
        context_cloud_ind = random.randint(0,len(clouds)-1)
        
        clusters = [torch_cluster.grid_cluster(x[:,:3], self.final_voxel_size) for x in clouds]
        
        valid_voxels = []
        for cluster in clusters:
            cluster_indices,counts = cluster.unique(return_counts=True)
            valid_indices = cluster_indices[counts>self.n_samples]
            valid_voxels.append(valid_indices)
        common_voxels = {}
        for ind,valid_voxel in enumerate(valid_voxels):
            if ind == context_cloud_ind: continue
            common_voxels[ind] = np.intersect1d(valid_voxels[context_cloud_ind],valid_voxel).tolist()
        valid_combs = []
        for key,val in common_voxels.items():
            valid_combs.extend([(key,x) for x in val])
        if len(valid_combs) < self.n_voxels:
            #If not enough recursively give other index from dataset
            print(f"Couldn't find combinations for index: {idx}")
            return self.__getitem__(random.randint(0,self.__len__()-1))
        draws = random.sample(valid_combs,self.n_voxels)


        voxel_points = []
        context_voxel_indices_list = []
        for draw in draws:
            cloud_ind = draw[0]
            indices  = clusters[cloud_ind]==draw[1]
            voxel = clouds[cloud_ind][indices,:]
            voxel = voxel[fps(voxel, torch.zeros(voxel.shape[0]).long(), ratio=self.n_samples/voxel.shape[0], random_start=False),:]
            voxel_points.append(voxel[:self.n_samples,:])
            context_voxel_indices = clusters[context_cloud_ind] ==draw[1]
            context_voxel = clouds[context_cloud_ind][context_voxel_indices,:]
            context_voxel_indices = context_voxel_indices.nonzero().squeeze()[fps(context_voxel, torch.zeros(context_voxel.shape[0]).long(), ratio=self.n_samples/context_voxel.shape[0], random_start=False)]
            context_voxel_indices_list.append(context_voxel_indices[:self.n_samples])





        #TODO Need to be smart about normalization after gnn
        #
        voxel_points = torch.stack(voxel_points)
        context_voxel_indices = torch.stack(context_voxel_indices_list)
        context = clouds[context_cloud_ind]
        
        return voxel_points,context,context_voxel_indices





