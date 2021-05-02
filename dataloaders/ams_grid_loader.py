import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import load_las, random_subsample,view_cloud_plotly,grid_split,co_min_max, circle_split,co_standardize,sep_standardize,unit_sphere,co_unit_sphere,extract_area
from itertools import permutations 
from torch_geometric.nn import fps
from tqdm import tqdm
from torch.utils.data import Dataset
import json
from datetime import datetime
from collections import Counter
eps = 1e-8
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
        self.path = os.path.join(base_dir,f'{self.id}.laz')
        self.datetime = datetime(int(self.recording_properties['RecordingTimeGps'].split('-')[0]),int(self.recording_properties['RecordingTimeGps'].split('-')[1]),int(self.recording_properties['RecordingTimeGps'].split('-')[-1].split('T')[0]))

class AmsGridLoader(Dataset):
    def __init__(self, directory_path,out_path,sample_size=2000,grid_square_size = 4,clearance = 10,preload=False,min_points=500,subsample='random',height_min_dif=0.5,normalization='min_max',grid_type='circle',max_height = 17.0, device="cuda"):
        self.sample_size  = sample_size
        self.directory_path = directory_path
        self.grid_square_size = grid_square_size
        self.clearance = clearance
        self.min_points = min_points
        self.out_path = out_path
        self.subsample = subsample
        self.height_min_dif = height_min_dif
        self.max_height = max_height
        self.minimum_difs = torch.Tensor([self.grid_square_size*0.95,self.grid_square_size*0.95,self.height_min_dif]).to(device)
        self.grid_type = grid_type
        self.save_name = f"ams_extract_id_dict_{grid_type}_{clearance}_{subsample}_{self.sample_size}_{self.min_points}_{self.grid_square_size}_{self.height_min_dif}.pt"
        self.normalization = normalization
        self.years = [2019,2020]
        with open(os.path.join(directory_path,'args.json')) as f:
            self.args = json.load(f)
        with open(os.path.join(directory_path,'response.json')) as f:
            self.response = json.load(f)
        
        

        if not preload:
            print(f"Recreating dataset, saving to: {self.out_path}")
            self.scans = [Scan(x,self.directory_path) for x in self.response['RecordingProperties']]
            self.scans = [x for x in self.scans if x.datetime.year in self.years]
            self.filtered_scans = filter_scans(self.scans,3)
            self.extract_id_dict = {}

            
            

            extract_id = -1
            for scene_number, scan in enumerate(tqdm(self.filtered_scans)):
                relevant_scans = [x for x in self.scans if np.linalg.norm(x.center-scan.center)<7]
                relevant_times = set([x.datetime for x in relevant_scans])
                time_partitions = {time:[x for x in relevant_scans if x.datetime ==time] for time in relevant_times}
                #centers_per_time = {key:sum([x.center for x in val])/len(val) for key,val in time_partitions.items()}
                
                clouds_per_time = [torch.from_numpy(np.concatenate([load_las(x.path) for x in val])).float().to(device) for key,val in time_partitions.items()]
                clouds_per_time = [x[x[:,2]<max_height,...] for x in clouds_per_time]
                if self.grid_type == 'square':
                    grids = [grid_split(cloud,self.grid_square_size,center=scan.center,clearance = self.clearance) for cloud in clouds_per_time]
                elif self.grid_type== "circle":
                    #Radius half of grid square size 
                    grids = [circle_split(cloud,self.grid_square_size/2,center=scan.center,clearance = self.clearance) for cloud in clouds_per_time]
                else:
                    raise Exception("Invalid grid type")
                

                    
                for square_index,extract_list in enumerate(list(zip(*grids))):
                    
                    extract_list = [x for x in extract_list if x.shape[0]>=self.min_points]
                    #Check mins
                    extract_list = [x for x in extract_list if ((x.max(dim=0)[0][:3]-x.min(dim=0)[0][:3] )>self.minimum_difs).all().item()]
                    
                    if len(extract_list)<2:
                        continue
                    
                    if self.subsample=='random':
                        extract_list = [ random_subsample(x,sample_size) for x in extract_list]
                    elif self.subsample=='fps':
                        extract_list = [ random_subsample(x,sample_size*5) for x in extract_list]
                        extract_list = [ x[fps(x,ratio = self.sample_size/x.shape[0])] if 0<self.sample_size/x.shape[0]<1 else x for x in extract_list]
                        
                    else:
                        raise Exception("Invalid subsampling type")
                    #Check mins again
                    extract_list = [x for x in extract_list if ((x.max(dim=0)[0][:3]-x.min(dim=0)[0][:3] )>self.minimum_difs).all().item()]

                    if len(extract_list)<2:
                        continue
                    extract_id +=1 # Iterate after continue to not skip ints

                    #Put on cpu before saving:
                    extract_list = [x.cpu() for x in extract_list]
                    for scan_index,extract in enumerate(extract_list):
                        
                        if not extract_id in self.extract_id_dict:
                            self.extract_id_dict[extract_id]=[]
                        self.extract_id_dict[extract_id].append(extract)
                        
            save_path  = os.path.join(self.out_path,self.save_name)
            print(f"Saving to {save_path}!")
            torch.save(self.extract_id_dict,save_path)
        else:
            self.extract_id_dict = torch.load(os.path.join(self.out_path,self.save_name))
        
        
        self.combinations_list=[]
        for id,path_list in self.extract_id_dict.items():
            index_permutations = list(permutations(range(len(path_list)),2))
            #Insert all unique permutations
            for perm in index_permutations:
                unique_combination = list(perm)
                unique_combination.insert(0,id)
                self.combinations_list.append(unique_combination)
            #Also include pairs with themselves
            for x in range(len(path_list)):
                self.combinations_list.append([id,x,x])


            
        print('Loaded dataset!')


    def __len__(self):
        return len(self.combinations_list)

    def view(self,index,point_size=5):
        cloud_1,cloud_2 = self.__getitem__(index)
        view_cloud_plotly(cloud_1[:,:3],cloud_1[:,3:],point_size=point_size)
        view_cloud_plotly(cloud_2[:,:3],cloud_2[:,3:],point_size=point_size)

    def test_nans(self):
        for i in range(self.__len__()):
            tensor_0, tensor_1 = self.__getitem__(i)
            if (tensor_0.isnan().any() or tensor_1.isnan().any()).item():
                raise Exception(f"Found nan at index {i}!")


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
        #Remove pesky extra points due to fps ratio
        tensor_0 = tensor_0[:self.sample_size,:]
        tensor_1 = tensor_1[:self.sample_size,:]
        tensor_0,tensor_1 = self.last_processing(tensor_0,tensor_1,self.normalization)

        
        
        return tensor_0,tensor_1


