from numpy.lib.function_base import extract
import torch
import os
import numpy as np
from utils import load_las, random_subsample,view_cloud_plotly,co_min_max,co_standardize,sep_standardize,extract_area,co_unit_sphere,get_voxel
from torch.utils.data import Dataset, DataLoader
from itertools import permutations 
from torch_geometric.nn import fps
from tqdm import tqdm
import pandas as pd

class ChallengeDataset(Dataset):
    def __init__(self, csv_path,direcories_list,out_path,voxel_size,sample_size=2000,radius = 1,preload=False,subsample='fps',normalization='co_unit_sphere',device="cuda",apply_normalization=True):
        self.sample_size  = sample_size
        self.out_path = out_path
        self.subsample = subsample
        self.radius = radius
        self.save_name = f"challenge_{self.subsample}_{self.sample_size}_{self.radius}.pt"
        self.normalization = normalization
        self.class_labels = ['nochange','removed',"added",'change',"color_change"]
        self.class_int_dict = {x:self.class_labels.index(x) for x in self.class_labels}
        self.int_class_dict = {val:key for key,val in self.class_int_dict.items()}
        self.apply_normalization = apply_normalization
        self.voxel_size = torch.tensor(voxel_size)
        
        if not preload :
            print(f"Recreating challenge dataset, saving to: {self.out_path}")
            df = pd.read_csv(csv_path)
            df = df[df['classification'].isin(self.class_labels)] #Remove unfit!
            scene_dicts = [{int(os.path.basename(x).split("_")[0]): os.path.join(year_path,x) for x in os.listdir(year_path) if x.split('.')[-1]=='las'} for year_path in direcories_list]
            combined_scene_dicts = {x:[scene_dicts[0][x],scene_dicts[1][x]] for x in scene_dicts[0].keys()}
            
            self.pair_dict={}
            
            pair_id = 0
            loaded_clouds = {}
            for index, row in tqdm(df.iterrows()):
                scene_num = row['scene']
                scene_path_list=combined_scene_dicts[scene_num]
                
                if not scene_num in loaded_clouds:
                    loaded_clouds[scene_num] = [torch.from_numpy(load_las(scene_path_list[x])).float().to(device) for x in range(2)]
                scene_0,scene_1 = loaded_clouds[scene_num]
                
                
                    
                label = self.class_int_dict[row['classification']]
                center = torch.Tensor([row['x'],row["y"]]).to(device)
                center = torch.cat((center,voxel_size[-1]/2))

                extract_0 = get_voxel(scene_0,center,voxel_size)
                extract_1 = get_voxel(scene_1,center,voxel_size)
                # extract_0 = scene_0[extract_area(scene_0,center,self.radius,'circle'),:]
                # extract_1 = scene_1[extract_area(scene_1,center,self.radius,'circle'),:]
                #Replace with placeholder if empty extract
                if extract_0.shape[0] == 0:
                    extract_0 = extract_1.mean(axis=0).unsqueeze(0)
                if extract_1.shape[0] == 0:
                    extract_1 = extract_0.mean(axis=0).unsqueeze(0)
                if subsample == 'random':
                    extract_0 = random_subsample(extract_0,sample_size)
                    extract_1 = random_subsample(extract_1,sample_size)
                elif subsample == "fps":
                    if self.sample_size/extract_0.shape[0]<1:
                        extract_0 = random_subsample(extract_0,sample_size*5)
                        extract_0 = extract_0[fps(extract_0,ratio = self.sample_size/extract_0.shape[0]),...]
                    if self.sample_size/extract_1.shape[0]<1:
                        extract_1= random_subsample(extract_1,sample_size*5)
                        extract_1 = extract_1[fps(extract_1,ratio = self.sample_size/extract_1.shape[0]),...]
                
                assert not (extract_0.isnan().any().item() or extract_1.isnan().any().item())
                torch_label = torch.Tensor([label]).long()
                self.pair_dict[pair_id]=[extract_0.float().cpu(),extract_1.float().cpu(),torch_label.cpu()]
                pair_id +=1
                


            save_path  = os.path.join(self.out_path,self.save_name)
            print(f"Saving to {save_path}!")
            torch.save(self.pair_dict,save_path)
        else:
            self.pair_dict = torch.load(os.path.join(self.out_path,self.save_name))
        print('Loaded dataset!')


    def __len__(self):
        return len(self.pair_dict)

    def view(self,index,point_size=3):
        extract_0,extract_1,label = self.pair_dict[index]
        label = self.int_class_dict[label.item()]
        print(f"This sample is classified as {label}!")
        view_cloud_plotly(extract_0[:,:3],extract_0[:,3:],point_size=point_size)
        view_cloud_plotly(extract_1[:,:3],extract_1[:,3:],point_size=point_size)

    def __getitem__(self, idx):
        
        extract_0,extract_1,label = self.pair_dict[idx]

        #Normalize
        if self.apply_normalization:
            if self.apply_normalization:
                if self.normalization == 'min_max':
                    extract_0[:,:3], extract_1[:,:3] = co_min_max(extract_0[:,:3],extract_1[:,:3])
                elif self.normalization == 'co_unit_sphere':
                    extract_0,extract_1 = co_unit_sphere(extract_0,extract_1)
                elif self.normalization == 'standardize':
                    extract_0,extract_1 = co_standardize(extract_0,extract_1)
                elif self.normalization == 'sep_standardize':
                    extract_0,extract_1 = sep_standardize(extract_0,extract_1)
                else:
                    raise Exception('Invalid normalization type')
        
        extract_0 = extract_0[:,:2000]
        extract_1 = extract_1[:,:2000]
        return extract_0,extract_1,label,idx