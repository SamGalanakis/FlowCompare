import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import load_las, random_subsample,view_cloud_plotly,grid_split,extract_area,co_min_max,feature_assigner
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from torch_geometric.data import Data,Batch
from torch_geometric.nn import fps
from dataloaders import ConditionalDataGrid, ShapeNetLoader
import wandb
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
import torch.distributed as distributed
from models import (
Full_matrix_combiner,
Exponential_combiner,
Learned_permuter,
conditional_exponential_matrix_coupling,
GCNEncoder,
ConditionalDenseNN, 
DenseNN,
Pointnet2
)



from torch_geometric.nn import DataParallel as geomDataParallel
from torch import nn

def main(rank, world_size):




    dirs = [r'/media/nfs/2_raid/sam/data_test/2018',r'/media/nfs/2_raid/sam/data_test/2019',r"/media/nfs/2_raid/sam/data_test/2020"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    config_path = r"config/config_conditional.yaml"
    wandb.init(project="flow_change",config = config_path)
    config = wandb.config
   
    torch.backends.cudnn.benchmark = True
    
    def collate_grid(batch):
        
        extract_0 = [item[0][:config['sample_size'],:config["input_dim"]] for item in batch]
        extract_1 = [item[1][:config['sample_size'],:config["input_dim"]] for item in batch]
        
        
        extract_1 = torch.stack(extract_1)
        return [extract_0, extract_1]

  






    one_up_path = os.path.dirname(__file__)
    out_path = os.path.join(one_up_path,r"save/processed_dataset")

    if config['data_loader'] == 'ConditionalDataGridSquare':
        dataset=ConditionalDataGrid(dirs,out_path=out_path,preload=config['preload'],subsample=config["subsample"],sample_size=config["sample_size"],min_points=config["min_points"],grid_type='square',normalization=config['normalization'],grid_square_size=config['grid_square_size'])
    elif config['data_loader'] == 'ConditionalDataGridCircle':
        dataset=ConditionalDataGrid(dirs,out_path=out_path,preload=config['preload'],subsample=config['subsample'],sample_size=config['sample_size'],min_points=config['min_points'],grid_type='circle',normalization=config['normalization'],grid_square_size=config['grid_square_size'])
    elif config['data_loader']=='ShapeNet':
        dataset = ShapeNetLoader(r'D:\data\ShapeNetCore.v2.PC15k\02691156\train',out_path=out_path,preload=config['preload'],subsample=config['subsample'],sample_size=config['sample_size'])
    else:
        raise Exception('Invalid dataloader type!')

    shuffle=True
    #SET PIN MEM TRUE

    dataloader = DataLoader(dataset,shuffle=shuffle,batch_size=config["batch_size"],num_workers=config['num_workers'],collate_fn=collate_grid,pin_memory=True,prefetch_factor=2)




    base_dist = dist.Normal(torch.zeros(config["input_dim"]).to(device), torch.ones(config["input_dim"]).to(device))


    
    




      
    class Conditional_flow_layers:
        def __init__(self,flow,n_flow_layers,input_dim,device,permuter,batchnorm):
            self.transformations = []
            self.n_flow_layers = n_flow_layers
            self.batchnorm = batchnorm
            
            for i in range(n_flow_layers):
                flow_module = flow()
                
                
                self.transformations.append(flow_module)
                
                if i<(self.n_flow_layers-1): #Don't add at the end
                    
                    #Batchnorm > permute > flow layer and end with flow layer
                    
  
                    permuter_instance = permuter()
            
                    self.transformations.append(permuter_instance)

                    if self.batchnorm:
                        bn_layer = BatchNorm(input_dim)
                        self.transformations.append(bn_layer)
                    
            self.layer_name_list = [type(x).__name__ for x in self.transformations]
        def make_save_list(self):
            save_list = []
            for x in self.transformations:
                try:
                    save_list.append(x.state_dict())
                    continue
                except:
                    pass
                try:
                    save_list.append(x.permutation)
                    continue
                except:
                    pass
                raise Exception('Can not save object')
            return save_list

        def to(self,device):
            for transform in self.transformations:
                try:
                    transform = transform.to(device)
                except:
                    continue

    if config["flow_type"] == 'exponential_coupling':
        flow = lambda  : conditional_exponential_matrix_coupling(input_dim=config["input_dim"], context_dim=config["context_dim"], hidden_dims=config["hidden_dims"], split_dim=None, dim=-1,device='cpu')
    elif config["flow_type"] == 'spline_coupling':
        flow = lambda : T.conditional_spline(input_dim=config["input_dim"], context_dim=config["context_dim"], hidden_dims=config["hidden_dims"],count_bins=config["count_bins"],bound=3.0)
    elif config["flow_type"] == 'spline_autoregressive':
        flow = lambda : T.conditional_spline_autoregressive(input_dim=config["input_dim"], context_dim=config["context_dim"], hidden_dims=config["hidden_dims"],count_bins=config["count_bins"],bound=3)
    elif config["flow_type"] == 'affine_coupling':
        flow = lambda : T.conditional_affine_coupling(input_dim=config["input_dim"], context_dim=config["context_dim"], hidden_dims=config["hidden_dims"])
    else:
        raise Exception(f'Invalid flow type: {config["flow_type"]}')
    if config["permuter_type"] == 'Exponential_combiner':
        permuter = lambda : Exponential_combiner(config["input_dim"])
    elif config["permuter_type"] == 'Learned_permuter':
        permuter = lambda : Learned_permuter(config["input_dim"])
    elif config["permuter_type"] == 'Full_matrix_combiner':
        permuter = lambda : Full_matrix_combiner(config["input_dim"])
    elif config["permuter_type"] == "random_permute":
        permuter = lambda : T.Permute(torch.randperm(config["input_dim"], dtype=torch.long).to(device))
    else:
        raise Exception(f'Invalid permuter type: {config["permuter_type"]}')



    
    
    conditional_flow_layers = Conditional_flow_layers(flow,config['n_flow_layers'],config['input_dim'],device,permuter,config['batchnorm'])

    
    parameters=[]

    if config["encoder_type"] == 'pointnet2':
        encoder = Pointnet2(feature_dim=config["input_dim"]-3,out_dim=config["context_dim"])
    elif config["encoder_type"] == 'gcn':
        encoder = GCNEncoder(in_dim= config["input_dim"],out_channels=config["context_dim"],k=20)
    else:
        raise Exception('Invalid encoder type!')
    if config["data_parallel"]:
        encoder = geomDataParallel(encoder).to(device)
    else:
        encoder = encoder.to(device)
    
    parameters+= encoder.parameters()
    wandb.watch(encoder,log_freq=10)

    if config["batchnorm_encodings"]:
        batchnorm_encoder = torch.nn.BatchNorm1d(config["context_dim"])
        if config["data_parallel"]:
            batchnorm_encoder = nn.DataParallel(batchnorm_encoder).to(device)
        else:
            batchnorm_encoder = batchnorm_encoder.to(device)
        parameters+= batchnorm_encoder.parameters()
    

    transformations = conditional_flow_layers.transformations
    

    for transform in transformations:
        if isinstance(transform,torch.nn.Module):
            transform.train()
            if config["data_parallel"]:
                transform = nn.DataParallel(transform).to(device)
            else:
                transform = transform.to(device)
            parameters+= transform.parameters()
            wandb.watch(transform,log_freq=10)



    flow_dist = dist.ConditionalTransformedDistribution(base_dist, transformations)
    
    if config["optimizer_type"] =='Adam':
        optimizer = torch.optim.Adam(parameters, lr=config["lr"],weight_decay=config["weight_decay"]) 
    elif config["optimizer_type"] == 'Adamax':
        optimizer = torch.optim.Adamax(parameters, lr=config["lr"])
    elif config["optimizer_type"] == 'AdamW':
        optimizer = torch.optim.AdamW(parameters, lr=config["lr"],weight_decay=config["weight_decay"])
    else:
        raise Exception('Invalid optimizer type!')

    

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=config["patience"],threshold=0.0001,min_lr=0.00005)
    save_model_path = r'save/conditional_flow_compare'
    

    
    def numpy_samples(conditioned,data_list_0):
        with torch.no_grad():
            sample = conditioned.sample([config["batch_size"],2000])[0].cpu().numpy()
            extract_0_0 = data_list_0[0]
            if config["input_dim"]>3:
                cond_nump = torch.cat((extract_0_0.pos,extract_0_0.x),dim=-1).cpu().numpy()
            else:
                cond_nump = torch.cat((extract_0_0.pos,torch.zeros_like(extract_0_0.pos))).cpu().numpy()
                sample = np.concatenate((sample,np.zeros_like(sample)),axis=-1)
            cond_nump[:,3:6] = np.clip(cond_nump[:,3:6]*255,0,255)
            sample[:,3:6] = np.clip(sample[:,3:6]*255,0,255)
            return cond_nump,sample
     


    torch.autograd.set_detect_anomaly(False)
    for epoch in range(config["n_epochs"]):
        print(f"Starting epoch: {epoch}")
        for batch_ind,batch in enumerate(tqdm(dataloader)):
            

            optimizer.zero_grad()
            extract_0,extract_1 = batch
            extract_0 =[x.to(device) for x in extract_0]
            data_list_0 = [Data(x=feature_assigner(x,config["input_dim"]),pos=x[:,:3]) for x in extract_0]
            extract_1 = extract_1.to(device)
            
            encodings = encoder(Batch.from_data_list(data_list_0))
            if config["batchnorm_encodings"]:
                encodings = batchnorm_encoder(encodings)
            assert not encodings.isnan().any(), "Nan in encoder"
            conditioned = flow_dist.condition(encodings.unsqueeze(-2))
            
           
            loss = -conditioned.log_prob(extract_1).mean()


            


            assert not loss.isnan(), "Nan loss!"
     
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters,max_norm=2.0)
            
            optimizer.step()
            
            flow_dist.clear_cache()
            
            scheduler.step(loss)
            current_lr = optimizer.param_groups[0]['lr']
            if batch_ind!=0 and  (batch_ind % int(len(dataloader)/100)  == 0):
                print(f'Making samples and saving!')
                with torch.no_grad():
                    cond_nump,gen_sample = numpy_samples(conditioned,data_list_0)
                    wandb.log({'loss':loss.item(),"Cond_cloud": wandb.Object3D(cond_nump),"Gen_cloud": wandb.Object3D(gen_sample),'lr':current_lr})
                    
                    if (batch_ind % int(len(dataloader)/5)  == 0):
                        save_dict = {"optimizer_dict": optimizer.state_dict(),'encoder_dict':encoder.state_dict(),'flow_transformations':conditional_flow_layers.make_save_list()}
                        torch.save(save_dict,os.path.join(save_model_path,f"{epoch}_{batch_ind}_model_dict.pt"))

            else:
                wandb.log({'loss':loss.item(),'lr':current_lr})
                
            
            
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    rank=''
    main(rank,world_size)