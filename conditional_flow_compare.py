import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import load_las, random_subsample,view_cloud_plotly,Early_stop,grid_split
from pyro.nn import DenseNN
from torch.utils.data import Dataset, DataLoader
from models.point_encoders import PointnetEncoder
from itertools import permutations 
from tqdm import tqdm
from models.pytorch_geometric_pointnet2 import Pointnet2

dirs = [r'D:\data\cycloData\multi_scan\2018',r'D:\data\cycloData\multi_scan\2020']

class ConditionalData(Dataset):
    def __init__(self, direcories_list,out_path,sample_size=1000,grid_square_size = 4,clearance = 28,preload=False,min_points=300):
        self.sample_size  = sample_size
        self.grid_square_size = grid_square_size
        self.clearance = clearance
        self.min_points = min_points
        self.out_path = out_path
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
                    extract_id +=1
                    extract_list = [x for x in extract_list if x.shape[0]>self.min_points]
                    if len(extract_list)<2:
                        continue
                    extract_list = [ random_subsample(x,sample_size) for x in extract_list]
                    for scan_index,extract in enumerate(extract_list):
                        save_name = f"{extract_id}_{scene_number}_{square_index}_{scan_index}_scan.npy"
                        np.save(os.path.join(self.out_path,save_name),extract,allow_pickle=True)
        file_paths_list  = [os.path.join(self.out_path,x) for x in os.listdir(self.out_path) if x.split('.')[-1]=='npy']
        self.extract_id_dict = {}
        for file_path in file_paths_list:
            id = int(os.path.basename(file_path).split("_")[0])
            if not id in self.extract_id_dict:
                self.extract_id_dict[id]=[]
            self.extract_id_dict[id].append(file_path)
        self.combinations_list=[]
        for id,path_list in self.extract_id_dict.items():
            index_permutations = list(permutations(range(len(path_list)),2))
            for perm in index_permutations:
                unique_combination = list(perm)
                unique_combination.insert(0,id)
                self.combinations_list.append(unique_combination)


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
        relevant_paths = self.extract_id_dict[combination_entry[0]]
        path_0 = relevant_paths[combination_entry[1]]
        path_1 = relevant_paths[combination_entry[2]]

        extract_0 = np.load(path_0)
        extract_1 = np.load(path_1)
        overall_max = np.maximum(extract_0[:,:3].max(axis=0),extract_1[:,:3].max(axis=0))
        overall_min = np.minimum(extract_0[:,:3].min(axis=0),extract_1[:,:3].min(axis=0))
        extract_0[:,:3] = (extract_0[:,:3] - overall_min)/(overall_max-overall_min)
        extract_1[:,:3] = (extract_1[:,:3] - overall_min)/(overall_max-overall_min)

        return torch.from_numpy(extract_0),torch.from_numpy(extract_1)

def my_collate(batch):
    extract_0 = [item[0] for item in batch]
    extract_1 = [item[1] for item in batch]
    batch_id_0 = [torch.ones(x.shape[0],dtype=torch.long)*index for index,x in enumerate(extract_0)]
    batch_id_1 = [torch.ones(x.shape[0],dtype=torch.long)*index for index,x in enumerate(extract_1)]
    extract_0 = torch.cat(extract_0)
    extract_1 = torch.cat(extract_1)
    batch_id_0 = torch.cat(batch_id_0)
    batch_id_1 = torch.cat(batch_id_1)
    return [extract_0, batch_id_0, extract_1,batch_id_1]

dataset=ConditionalData(dirs,out_path="save//processed_dataset",preload=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloader = DataLoader(dataset,shuffle=True,batch_size=3,num_workers=0,prefetch_factor=2,collate_fn=my_collate)
Pointnet2 = Pointnet2()

for extract_0, batch_id_0, extract_1,batch_id_1 in dataloader:
    Pointnet2(extract_0[:,3:],extract_0[:,:3],batch_id_0)
    pass

input_dim = 6

base_dist = dist.Normal(torch.zeros(input_dim).to(device), torch.ones(input_dim).to(device))
count_bins = 16
param_dims = lambda split_dimension: [(input_dim - split_dimension) * count_bins,
(input_dim - split_dimension) * count_bins,
(input_dim - split_dimension) * (count_bins - 1),
(input_dim - split_dimension) * count_bins]

split_dims = [2]*3
patience = 50
n_blocks = 2
permutations =  [[1,0,2],[2,0,1],[2,1,0]]


class conditional_flow_block:
    def __init__(self,input_dim,permutations,count_bins,split_dims,device):
        self.transformations = []
        self.parameters =[]
        param_dims = lambda split_dimension: [(input_dim - split_dimension) * count_bins,
        (input_dim - split_dimension) * count_bins,
        (input_dim - split_dimension) * (count_bins - 1),
        (input_dim - split_dimension) * count_bins]
        for i, permutation in enumerate(permutations):
            hypernet =  DenseNN(split_dims[i], [10*input_dim], param_dims(split_dims[i]))
            spline = T.ConditionalSpline(input_dim = input_dim, split_dim = split_dims[i] , count_bins=count_bins,nn=hypernet)
            spline = spline.to(device)
            self.parameters += spline.parameters()
            self.transformations.append(spline)
            self.transformations.append(T.permute(input_dim,torch.LongTensor(permutations[i]).to(device),dim=-1))
    def save(self,path):
        torch.save(self,path)

flow_blocks = [flow_block(input_dim,permutations,count_bins,split_dims,device) for x in range(n_blocks)]

parameters = []
transformations = []
for flow_block_instance in flow_blocks:
    parameters.extend(flow_block_instance.parameters)
    transformations.extend(flow_block_instance.transformations)


flow_dist = dist.TransformedDistribution(base_dist, transformations)




steps = 3000
early_stop_margin=0.01
optimizer = torch.optim.AdamW(parameters, lr=5e-3) #was 5e-3

early_stop = Early_stop(patience=patience,min_perc_improvement=torch.tensor(early_stop_margin))

for step in range(steps+1):
    X = random_subsample(points1_scaled,n_samples)
    #dataset = torch.tensor(X, dtype=torch.float).to(device)
    dataset=X
    dataset += torch.randn_like(dataset)*0.01
    optimizer.zero_grad()
    loss = -flow_dist.log_prob(dataset).mean()
    
    
    
    
    last_loss = loss
    loss.backward()
    optimizer.step()
    flow_dist.clear_cache()

    stop_training = early_stop.log(loss.detach().cpu())

    if stop_training:
        print(f"Ran out of patience at step: {step}, Cur_loss: {loss}, Best: {early_stop.best_loss}")
        
        break

    
    

log_probs_1 = flow_dist.log_prob(points1_scaled).detach().cpu().numpy()
log_probs_2 = flow_dist.log_prob(points2_scaled).detach().cpu().numpy()


    
