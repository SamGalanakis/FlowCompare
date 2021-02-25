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
from torch_geometric.nn import voxel_grid


eps = 1e-8



class ConditionalVoxelGrid(Dataset):
    def __init__(self, direcories_list,out_path,voxel_size=0.1,grid_square_size = 4,clearance = 28,preload=False,min_points=500,in_vox_sample_size= 100,subsample='random',height_min_dif=0.5,normalization='min_max'):
        self.in_vox_sample_size  = in_vox_sample_size
        self.grid_square_size = grid_square_size
        self.clearance = clearance
        self.min_points = min_points
        self.out_path = out_path
        self.extract_id_dict = {}
        self.voxel_indicies_id_dict = {}
        self.subsample = subsample
        self.height_min_dif = height_min_dif
        self.minimum_difs = torch.Tensor([self.grid_square_size*0.95,self.grid_square_size*0.95,self.height_min_dif])
        self.save_name = f"voxel_dataset_{clearance}_{subsample}_{self.in_vox_sample_size}_{self.min_points}_{self.grid_square_size}{self.voxel_size}.pt"
        self.normalization = normalization
        self.vozel_size = voxel_size
        if not preload:
            print(f"Recreating dataset, saving to: {self.out_path}")
            file_path_lists  = [[os.path.join(path,x) for x in os.listdir(path) if x.split('.')[-1]=='las'] for path in direcories_list]
            scene_dict = {}
            
            for file_path_list in file_path_lists:
                source_dir_name = os.path.basename(os.path.dirname(file_path_list[0]))
                
                for path in file_path_list:
                    scene_number = int(os.path.basename(path).split("_")[0])
                    scan_number = int(os.path.basename(path).split("_")[1])
                    if not scene_number in scene_dict:
                        scene_dict[scene_number]=[]
                    scene_dict[scene_number].append(path)

            extract_id = -1
            for scene_number, path_list in tqdm(scene_dict.items()):
                full_clouds = [load_las(path) for path in path_list]
                center = full_clouds[0][:,:2].mean(axis=0)
                grids = [grid_split(cloud,self.grid_square_size,center=center,clearance = self.clearance) for cloud in full_clouds]
                

                for square_index,extract_list in enumerate(list(zip(*grids))):
                    
                    extract_list = [torch.from_numpy(x.astype(np.float32)) for x in extract_list if x.shape[0]>=self.min_points]
                    #Check mins
                    extract_list = [x for x in extract_list if ((x.max(dim=0)[0][:3]-x.min(dim=0)[0][:3] )>self.minimum_difs).all().item()]
                    if len(extract_list)<2:
                        continue
                    else:
                        # Iterate after check to not skip ints
                        extract_id +=1
    
                         

                    for scan_index,extract in enumerate(extract_list):
                        if not extract_id in self.extract_id_dict:
                            self.extract_id_dict[extract_id]=[]
                        self.extract_id_dict[extract_id].append(extract)

                        if not extract_id in self.voxel_indicies_id_dict:
                            self.voxel_indicies_id_dict[extract_id]=[]
                        self.voxel_indicies_id_dict[extract_id].append(voxel_indices[scan_index])
                        
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

    def view(self,index):
        cloud_1,cloud_2 = self.__getitem__(index)
        view_cloud_plotly(cloud_1[:,:3],cloud_1[:,3:])
        view_cloud_plotly(cloud_2[:,:3],cloud_2[:,3:])

    def test_nans(self):
        for i in range(self.__len__()):
            tensor_0, tensor_1 = self.__getitem__(i)
            if (tensor_0.isnan().any() or tensor_1.isnan().any()).item():
                raise Exception(f"Found nan at index {i}!")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        combination_entry = self.combinations_list[idx]
        relevant_tensors = self.extract_id_dict[combination_entry[0]]
        relevant_voxel_indices = self.voxel_indicies_id_dict[combination_entry[0]]
        #CLONE THE TENSOR IF SAME, OTHERWISE POINT TO SAME MEMORY, PROBLEMS IN NORMALIZATION
        if combination_entry[1]!=combination_entry[2]:
            tensor_0 = relevant_tensors[combination_entry[1]]
            tensor_1 = relevant_tensors[combination_entry[2]]
        else:
            tensor_0 = relevant_tensors[combination_entry[1]]
            tensor_1 = relevant_tensors[combination_entry[2]].clone()

        if self.normalization == 'min_max':
            tensor_0[:,:3], tensor_1[:,:3] = co_min_max(tensor_0[:,:3],tensor_1[:,:3])
        elif self.normalization == 'normalize':
            concatenated = torch.cat((tensor_0[:,:3],tensor_1[:,:3]),dim=0)
            concatenated = (concatenated-concatenated.mean(axis=0))/(concatenated.std()+eps)
            tensor_0[:,:3], tensor_1[:,:3] = torch.split(concatenated,tensor_0.shape[0],dim=0)
        else:
            raise Exception('Invalid normalization type')
        
        return tensor_0,tensor_1
