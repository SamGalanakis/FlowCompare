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



class ConditionalDataGrid(Dataset):
    def __init__(self, direcories_list,out_path,sample_size=2000,grid_square_size = 4,clearance = 28,preload=False,min_points=500,subsample='random'):
        self.sample_size  = sample_size
        self.grid_square_size = grid_square_size
        self.clearance = clearance
        self.min_points = min_points
        self.out_path = out_path
        self.extract_id_dict = {}
        self.subsample = subsample
        self.save_name = f"extract_id_dict_{subsample}_{self.sample_size}_{self.min_points}_{self.grid_square_size}.pt"
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
                    if len(extract_list)<2:
                        continue
                    extract_id +=1 # Iterate after continue to not skip ints\
                    if self.subsample=='random':
                        extract_list = [ random_subsample(x,sample_size) for x in extract_list]
                    elif self.subsample=='fps':
                        
                        extract_list = [ x[fps(x.cuda(),ratio = self.sample_size/x.shape[0])] if 0<self.sample_size/x.shape[0]<1 else x for x in extract_list]

                    for scan_index,extract in enumerate(extract_list):
                        #save_name = f"{extract_id}_{scene_number}_{square_index}_{scan_index}_scan.npy"
                        if not extract_id in self.extract_id_dict:
                            self.extract_id_dict[extract_id]=[]
                        self.extract_id_dict[extract_id].append(extract)
                        #np.save(os.path.join(self.out_path,save_name),extract,allow_pickle=True)
            save_path  = os.path.join(self.out_path,self.save_name)
            print(f"Saving to {save_path}!")
            torch.save(self.extract_id_dict,save_path)
        else:
            self.extract_id_dict = torch.load(os.path.join(self.out_path,self.save_name))
        
        #file_paths_list  = [os.path.join(self.out_path,x) for x in os.listdir(self.out_path) if x.split('.')[-1]=='npy']
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

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        combination_entry = self.combinations_list[idx]
        relevant_tensors = self.extract_id_dict[combination_entry[0]]
        tensor_0 = relevant_tensors[combination_entry[1]]
        tensor_1 = relevant_tensors[combination_entry[2]]
        tensor_0[:,:3], tensor_1[:,:3] = co_min_max(tensor_0[:,:3],tensor_1[:,:3])
        return tensor_0,tensor_1