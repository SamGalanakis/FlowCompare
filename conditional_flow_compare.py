import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import load_las, random_subsample,view_cloud_plotly,Early_stop,grid_split
from pyro.nn import DenseNN
from torch.utils.data import Dataset, DataLoader




dirs = [r'D:\data\cycloData\multi_scan\2018',r'D:\data\cycloData\multi_scan\2020']

class ConditionalData(Dataset):
    def __init__(self, direcories_list,out_path,sample_size=1000,grid_square_size = 4,clearance = 16):
        self.sample_size  = sample_size
        self.grid_square_size = grid_square_size
        self.clearance = clearance
        self.min_points = 100
        self.out_path = out_path
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


        for scene_number, path_list in scene_dict.items():
            full_clouds = [load_las(path) for path in path_list]
            center = full_clouds[0][:,:2].mean(axis=0)
            grids = [grid_split(cloud,self.grid_square_size,center=center,clearance = self.clearance) for cloud in full_clouds]

            for square_index,extract_list in enumerate(list(zip(*grids))):
                extract_list = [x for x in extract_list if x.shape[0]>self.min_points]
                if len(extract_list)<2:
                    continue
                extract_list = [ random_subsample(x,sample_size) for x in extract_list]
                for scan_index,extract in enumerate(extract_list):
                    save_name = f"{scene_number}_{square_index}_{scan_index}_scan.npy"
                    np.save(os.path.join(self.out_path,save_name),extract,allow_pickle=True)





            



        


        
    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


ConditionalData(dirs,out_path="save//processed_dataset")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_samples=2000 #was 1000
input_dim=3
points1 = points1[:,:input_dim]
points2 = points2[:,:input_dim] 
scaler = MinMaxScaler()
scaler.fit(np.concatenate((points1,points2),axis=0))

points1_scaled = scaler.transform(points1)
points1_scaled = torch.tensor(points1_scaled, dtype=torch.float).to(device)
points2_scaled = scaler.transform(points2)
points2_scaled = torch.tensor(points2_scaled, dtype=torch.float).to(device)

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


class flow_block:
    def __init__(self,input_dim,permutations,count_bins,split_dims,device):
        self.transformations = []
        self.parameters =[]
        param_dims = lambda split_dimension: [(input_dim - split_dimension) * count_bins,
        (input_dim - split_dimension) * count_bins,
        (input_dim - split_dimension) * (count_bins - 1),
        (input_dim - split_dimension) * count_bins]
        for i, permutation in enumerate(permutations):
            hypernet =  DenseNN(split_dims[i], [10*input_dim], param_dims(split_dims[i]))
            spline = T.SplineCoupling(input_dim = input_dim, split_dim = split_dims[i] , count_bins=count_bins,hypernet=hypernet)
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


    
