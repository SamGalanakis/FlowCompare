

def main():


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
    from itertools import permutations, combinations
    from tqdm import tqdm
    from models.pytorch_geometric_pointnet2 import Pointnet2
    from torch_geometric.nn import fps
    from dataloaders.ConditionalDataGrid import ConditionalDataGrid
    import wandb

    dirs = [r'D:\data\cycloData\multi_scan\2018',r'D:\data\cycloData\multi_scan\2020']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = "config/config_conditional.yaml"
    wandb.init(project="flow_change",config = config_path)

    sample_size= wandb.config['sample_size'] 
    n_flow_layers = wandb.config['n_flow_layers']
    early_stop_margin = wandb.config['early_stop_margin']
    save_model_path = wandb.config['save_model_path']
    count_bins = wandb.config['count_bins']
    input_dim = wandb.config['input_dim']
    batch_size = wandb.config['batch_size']
    grid_square_size = wandb.config['grid_square_size']
    clearance = wandb.config['clearance']
    subsample = wandb.config['subsample']
    patience = wandb.config['patience']
    preload = wandb.config['preload']
    min_points = wandb.config['min_points']
    n_epochs = wandb.config['n_epochs']
    context_dim = wandb.config['context_dim']
    lr = wandb.config['lr']
    num_workers = wandb.config['num_workers']




    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def my_collate(batch):
        extract_0 = [item[0] for item in batch]
        extract_1 = [item[1] for item in batch]
        batch_id_0 = [torch.ones(x.shape[0],dtype=torch.long)*index for index,x in enumerate(extract_0)]
        extract_0 = torch.cat(extract_0)
        extract_1 = torch.stack(extract_1)
        batch_id_0 = torch.cat(batch_id_0)
        return [extract_0, batch_id_0, extract_1]


    dataset=ConditionalDataGrid(dirs,out_path="save//processed_dataset",preload=preload,subsample=subsample,sample_size=sample_size,min_points=min_points)
    
    if num_workers>0:
        torch.multiprocessing.freeze_support()
      
    dataloader = DataLoader(dataset,shuffle=True,batch_size=batch_size,num_workers=num_workers,collate_fn=my_collate,pin_memory=True)




    base_dist = dist.Normal(torch.zeros(input_dim).to(device), torch.ones(input_dim).to(device))


    Pointnet2 = Pointnet2(feature_dim=input_dim-3,out_dim=context_dim).to(device)
    permutations = [torch.randperm(input_dim) for x in range(n_flow_layers-1)]


    class conditional_spline_flow:
        def __init__(self,input_dim,context_dim,permutations,count_bins,device):
            self.transformations = []
            self.parameters =[]
            
            for i in range(len(permutations)+1):
                hidden_dims = [128,128]
                spline = T.conditional_spline(input_dim,context_dim,hidden_dims=hidden_dims,count_bins=count_bins,bound=1.0)
                spline = spline.to(device)
                self.parameters += spline.parameters()
                self.transformations.append(spline)
                if i<len(permutations): #Not try to add to the end
                    self.transformations.append(T.permute(input_dim,torch.LongTensor(permutations[i]).to(device),dim=-1))
        def save(self,path):
            torch.save(self,path)

    conditional_flow_layers = conditional_spline_flow(input_dim,context_dim,permutations,count_bins,device)


    parameters = conditional_flow_layers.parameters
    parameters+= Pointnet2.parameters()
    transformations = conditional_flow_layers.transformations



    flow_dist = dist.ConditionalTransformedDistribution(base_dist, transformations)


    optimizer = torch.optim.AdamW(parameters, lr=lr) 


    full_model = {'Encoder':Pointnet2,'transformations':transformations}

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(n_epochs):
        print(f"Starting epoch: {epoch}")
        for batch in tqdm(dataloader):
            
            optimizer.zero_grad(set_to_none=True)
            extract_0,enumeration_0,extract_1 = batch
            extract_0 = extract_0.to(device)
            enumeration_0 = enumeration_0.to(device)
            extract_1 = extract_1.to(device)
            
            encodings = Pointnet2(extract_0[:,3:],extract_0[:,:3],enumeration_0)
            with torch.cuda.amp.autocast(): #Encoder don't work with this.
                loss=0
                for batch_ind in range(extract_1.shape[0]):
                    encoding = encodings[batch_ind,:]
                    extract_1_points = extract_1[batch_ind,...]
                    loss += -flow_dist.condition(encoding).log_prob(extract_1_points).mean()
        
            scaler.scale(loss).backward()

        
            
           
            scaler.step(optimizer)
            scaler.update()
            flow_dist.clear_cache()
            wandb.log({'loss':loss.item()})
        torch.save(full_model,os.path.join(save_model_path,f"{epoch}_model_dict.pt"))
if __name__ == "__main__":
    main()

       
            
        

    



    
