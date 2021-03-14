from numpy.lib.function_base import extract
import torch
import os
import numpy as np
from utils import load_las, random_subsample,view_cloud_plotly,co_min_max,co_standardize,sep_standardize,extract_area,rgb_to_hsv,ground_remover
from torch.utils.data import Dataset, DataLoader
from itertools import permutations 
from torch_geometric.nn import fps
from tqdm import tqdm
import pandas as pd


class ChallengeDataset(Dataset):
    def __init__(self, csv_path,direcories_list,out_path,sample_size=2000,radius = 4,preload=False,subsample='random',normalization='min_max',device="cuda",subset= None,apply_normalization=True,remove_ground=True):
        self.sample_size  = sample_size
        self.out_path = out_path
        self.subsample = subsample
        self.radius = radius
        self.save_name = f"challenge_{self.subsample}_{self.sample_size}_{self.radius}.pt"
        self.normalization = normalization
        self.class_labels = ['nochange','removed',"added",'change',"color_change"]
        self.class_int_dict = {x:self.class_labels.index(x) for x in self.class_labels}
        self.int_class_dict = {val:key for key,val in self.class_int_dict.items()}
        self.subset = subset
        self.apply_normalization = apply_normalization
        self.remove_ground = remove_ground
        if not preload:
            print(f"Recreating challenge dataset, saving to: {self.out_path}")
            csv_path_list = [os.path.join(csv_path,x) for x in os.listdir(csv_path) if x.split('.')[-1]=='csv']
            scene_path_dict = {int(os.path.basename(x).split("_")[0]): x for x in csv_path_list}
            scene_df_dict = {int(os.path.basename(x).split("_")[0]): pd.read_csv(x) for x in csv_path_list}
            file_path_lists  = [[os.path.join(path,x) for x in os.listdir(path) if x.split('.')[-1]=='las'] for path in direcories_list]
            scene_dicts = [{int(os.path.basename(x).split("_")[0]): os.path.join(year_path,x) for x in os.listdir(year_path) if x.split('.')[-1]=='las'} for year_path in direcories_list]
            combined_scene_dicts = {x:[scene_dicts[0][x],scene_dicts[1][x]] for x in scene_dicts[0].keys()}
            
            self.pair_dict={}
            
            pair_id = 0
            for scene_number, df in tqdm(sorted(scene_df_dict.items())):
                scene_path_list=combined_scene_dicts[scene_number]
                scene_0 = load_las(scene_path_list[0])
                scene_1 = load_las(scene_path_list[1])
                for index,row in df.iterrows():
                    
                    label = self.class_int_dict[row['classification']]
                    center = np.array([row['x'],row["y"]])

                    
                    extract_0 = torch.from_numpy(scene_0[extract_area(scene_0,center,self.radius,'circle'),:]).to(device)
                    extract_1 = torch.from_numpy(scene_1[extract_area(scene_1,center,self.radius,'circle'),:]).to(device)
                    #Replace with placeholder if empty extract
                    if extract_0.shape[0] == 0:
                        extract_0 = extract_1.mean(axis=0).unsqueeze(0)
                    if extract_1.shape[0] == 0:
                        extract_0 = extract_0.mean(axis=0).unsqueeze(0)
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
                    extract_0[:,3:] = rgb_to_hsv(extract_0[:,3:])
                    extract_1[:,3:] = rgb_to_hsv(extract_1[:,3:])
                    assert not (extract_0.isnan().any().item() or extract_1.isnan().any().item())
                    torch_label = torch.Tensor([label]).long()
                    self.pair_dict[pair_id]=[extract_0.float().cpu(),extract_1.float().cpu(),torch_label.cpu()]
                    pair_id +=1
                    


            save_path  = os.path.join(self.out_path,self.save_name)
            print(f"Saving to {save_path}!")
            torch.save(self.pair_dict,save_path)
        else:
            self.pair_dict = torch.load(os.path.join(self.out_path,self.save_name))
        
        if subset!=None:
            assert all([x in self.pair_dict.keys() for x in subset]), "Invalid subset"
            print(f"Using subset: {self.subset}")
            self.pair_dict = {key:val for key,val in self.pair_dict.items() if key in self.subset}
            self.subset_map = {x:sorted(self.subset)[x] for x in range(len(self.subset))}
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
        if self.subset!=None:
            idx = self.subset_map[idx]
        extract_0,extract_1,label = self.pair_dict[idx]
        if self.apply_normalization:
            if self.apply_normalization:
                if self.normalization == 'min_max':
                    extract_0[:,:3], extract_1[:,:3] = co_min_max(extract_0[:,:3],extract_1[:,:3])
                elif self.normalization == 'standardize':
                    extract_0,extract_1 = co_standardize(extract_0,extract_1)
                elif self.normalization == 'sep_standardize':
                    extract_0,extract_1 = sep_standardize(extract_0,extract_1)
                else:
                    raise Exception('Invalid normalization type')
        if self.remove_ground:
            extract_0 = extract_0[ground_remover(extract_0),:]
            extract_1 = extract_1[ground_remover(extract_1),:]
        return extract_0,extract_1,label,idx