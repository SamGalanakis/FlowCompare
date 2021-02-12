def main():


    import torch
    import pyro
    import pyro.distributions as dist
    import pyro.distributions.transforms as T
    import os
    import numpy as np
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from utils import load_las, random_subsample,view_cloud_plotly,Early_stop,grid_split
    from pyro.nn import DenseNN
    from torch.utils.data import Dataset, DataLoader
    from itertools import permutations, combinations
    from tqdm import tqdm
    from models.pytorch_geometric_pointnet2 import Pointnet2
    from torch_geometric.nn import fps
    from dataloaders.ConditionalDataGrid import ConditionalDataGrid
    import wandb
    import torch.multiprocessing as mp
    from torch.nn.parallel import DataParallel
    import torch.distributed as distributed
    from torch_geometric.nn import DataParallel as geomDataParallel
    from models.permuters import Full_matrix_combiner,Exponential_combiner
    from models.scalers import Sigmoid_scaler
    from models.batchnorm import BatchNorm
    from torch.autograd import Variable, Function
    from models.Exponential_matrix_flow import conditional_exponential_matrix_coupling
    


    dirs = [r'/mnt/cm-nas03/synch/students/sam/data_test/2018',r'/mnt/cm-nas03/synch/students/sam/data_test/2019',r'/mnt/cm-nas03/synch/students/sam/data_test/2020']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    config_path = r"config/config_conditional_light.yaml"
    wandb.init(project="flow_change",config = config_path)
    config = wandb.config
    sample_size= config['sample_size'] 
    n_flow_layers = config['n_flow_layers']
    early_stop_margin = config['early_stop_margin']
    hidden_dims = config['hidden_dims']
    save_model_path = config['save_model_path']
    count_bins =config['count_bins']
    input_dim = config['input_dim']
    batch_size = wandb.config['batch_size']
    grid_square_size = config['grid_square_size']
    clearance = config['clearance']
    subsample = config['subsample']
    patience = config['patience']
    preload = config['preload']
    min_points = config['min_points']
    n_epochs = config['n_epochs']
    context_dim = config['context_dim']
    lr = config['lr']
    num_workers = config['num_workers']
    permuter_type = config['permuter_type']
    scaler_type = config['scaler_type']
    flow_type = config['flow_type']
    batchnorm = config['batchnorm']


    torch.backends.cudnn.benchmark = True
    
    def my_collate(batch):
        extract_0 = [item[0][:,:input_dim] for item in batch]
        extract_1 = [item[1][:,:input_dim] for item in batch]
        batch_id_0 = [torch.ones(x.shape[0],dtype=torch.long)*index for index,x in enumerate(extract_0)]
        extract_0 = torch.cat(extract_0)
        extract_1 = torch.stack(extract_1)
        batch_id_0 = torch.cat(batch_id_0)
        if (extract_0.isnan().any() or extract_1.isnan().any()).item():
            print()
        return [extract_0, batch_id_0, extract_1]

  






    one_up_path = os.path.dirname(__file__)
    out_path = os.path.join(one_up_path,"save/processed_dataset")
    dataset=ConditionalDataGrid(dirs,out_path=out_path,preload=preload,subsample=subsample,sample_size=sample_size,min_points=min_points)
 

    
    shuffle=True
    dataloader = DataLoader(dataset,shuffle=shuffle,batch_size=batch_size,num_workers=num_workers,collate_fn=my_collate,pin_memory=True,prefetch_factor=2)




    base_dist = dist.Normal(torch.zeros(input_dim).to(device), torch.ones(input_dim).to(device))


    





        
    #NEED TO DEAL WITH THE ORDER OF TRANSFORMATIONS AS TRAINING USED THE INVERS!!!
    class Conditional_flow_layers:
        def __init__(self,flow,n_flow_layers,input_dim,context_dim,count_bins,device,permuter,scaler,hidden_dims,batchnorm):
            self.transformations = []
            self.parameters =[]
            self.n_flow_layers = n_flow_layers
            self.hidden_dims = hidden_dims
            self.batchnorm = batchnorm
            
            for i in range(n_flow_layers):
                flow_module = flow().to(device)
                
                self.parameters += flow_module.parameters()
                self.transformations.append(flow_module)
                
                if i<(self.n_flow_layers-1): #Don't add at the end
                    
                    if self.batchnorm:
                        bn_layer = BatchNorm(input_dim).to(device)
                        self.parameters += bn_layer.parameters()
                        self.transformations.append(bn_layer)
                    
                    #self.transformations.append(scaler())
                    permuter_instance = permuter()
                    try:
                        permuter_instance = permuter_instance.to(device)
                        self.parameters += permuter_instance.parameters()
                    except:
                        pass
                    self.transformations.append(permuter_instance)
                    
            self.layer_name_list = [type(x).__name__ for x in self.transformations]
        def save(self,path):
            torch.save(self,path)

    if flow_type == 'exponential_coupling':
        flow = lambda  : conditional_exponential_matrix_coupling(input_dim=input_dim, context_dim=context_dim, hidden_dims=hidden_dims, split_dim=None, dim=-1,device=device)
    elif flow_type == 'spline_coupling':
        flow = lambda : T.conditional_spline(input_dim=input_dim, context_dim=context_dim, hidden_dims=hidden_dims,count_bins=count_bins,bound=3.0)
    elif flow_type == 'spline_autoregressive':
        flow = lambda : T.conditional_spline_autoregressive(input_dim=input_dim, context_dim=context_dim, hidden_dims=hidden_dims,count_bins=count_bins,bound=3)
    else:
        raise Exception(f'Invalid flow type: {flow_type}')
    if permuter_type == 'Exponential_combiner':
        permuter = lambda : Exponential_combiner(input_dim)
    elif permuter_type == 'Full_matrix_combiner':
        permuter = lambda : Full_matrix_combiner(input_dim)
    elif permuter_type == "random_permute":
        permuter = lambda : T.Permute(torch.randperm(input_dim, dtype=torch.long).to(device))
    else:
        raise Exception(f'Invalid permuter type: {permuter_type}')



    if scaler_type == 'Sigmoid_scaler':
        scaler = Sigmoid_scaler
    
    
    conditional_flow_layers = Conditional_flow_layers(flow,n_flow_layers,input_dim,context_dim,count_bins,device,permuter,scaler,hidden_dims,batchnorm)

    
    parameters = conditional_flow_layers.parameters


    Pointnet2 = Pointnet2(feature_dim=input_dim-3,out_dim=context_dim)
    Pointnet2 = Pointnet2.to(device).train()
    wandb.watch(Pointnet2,log='gradients',log_freq=10)


    parameters+= Pointnet2.parameters()
    transformations = conditional_flow_layers.transformations
    
    for transform in transformations:
        if isinstance(transform,torch.nn.Module):
            transform.train()
            wandb.watch(transform,log='gradients',log_freq=10)



    flow_dist = dist.ConditionalTransformedDistribution(base_dist, transformations)


    optimizer = torch.optim.AdamW(parameters, lr=lr) 


    save_model_path = r'save/conditional_flow_compare'
    


    torch.autograd.set_detect_anomaly(False)
    for epoch in range(n_epochs):
        print(f"Starting epoch: {epoch}")
        for batch_ind,batch in enumerate(tqdm(dataloader)):
            
            optimizer.zero_grad()
            extract_0,enumeration_0,extract_1 = batch
            if (extract_0.isnan().any() or extract_1.isnan().any()).item():
                print('Found nan, skipping batch!')
                continue
     
            extract_0 = extract_0.to(device)
            enumeration_0 = enumeration_0.to(device)
            extract_1 = extract_1.to(device)
            encodings = Pointnet2(extract_0[:,3:],extract_0[:,:3],enumeration_0) 
            assert not encodings.isnan().any(), "Nan in encoder"
            conditioned = flow_dist.condition(encodings.unsqueeze(-2))
            loss = -conditioned.log_prob(extract_1).mean()

        
           

        
            assert not loss.isnan(), "Nan loss!"
            loss.backward()
            torch.nn.utils.clip_grad_value_(parameters,10)
           
            optimizer.step()
           
            flow_dist.clear_cache()
            wandb.log({'loss':loss.item()})
            if batch_ind!=0 and  (batch_ind % int(len(dataloader)/10 +1)  == 0) :
                save_dict = {"optimizer_dict": optimizer.state_dict(),'encoder_dict':Pointnet2.state_dict(),'flow_transformations':transformations}
                torch.save(save_dict,os.path.join(save_model_path,f"{epoch}_{batch_ind}_model_dict.pt"))
if __name__ == "__main__":
    main()