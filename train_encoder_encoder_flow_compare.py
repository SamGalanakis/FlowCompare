import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import load_las, random_subsample,view_cloud_plotly,grid_split,extract_area,co_min_max,feature_assigner,Adamax
from torch.utils.data import Dataset,DataLoader
from itertools import permutations, combinations
from tqdm import tqdm
from torch_geometric.data import Data,Batch
from torch_geometric.nn import fps
from dataloaders import ConditionalDataGrid, ShapeNetLoader
import wandb
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
import torch.distributed as distributed
from torch.autograd import Variable, Function
import torch.multiprocessing as mp
from torch_geometric.nn import DataParallel as geomDataParallel
from torch import nn
import functools
from models import (
Full_matrix_combiner,
Exponential_combiner,
Learned_permuter,
conditional_exponential_matrix_coupling,
GCNEncoder,
ConditionalDenseNN, 
DenseNN,
Pointnet2,
Conditional_flow_layers
)


def load_transformations(load_dict,conditional_flow_layers):
    for transformation_params,transformation in zip(load_dict['flow_transformations'],conditional_flow_layers.transformations):
        if isinstance(transformation,nn.Module):
            transformation.load_state_dict(transformation_params)
        elif isinstance(transformation,pyro.distributions.pyro.distributions.transforms.Permute):
            transformation.permutation = transformation_params
        else:
            raise Exception('How to load?')
    
    return conditional_flow_layers



def initialize_encoder_models(config,device = 'cuda',mode='train'):
    flow_input_dim = config['context_dim']
    flow_type = config['flow_type']
    permuter_type = config['permuter_type']
    hidden_dims = config['hidden_dims']
    data_parallel = config['data_parallel']
    parameters = []

    if config['coupling_block_nonlinearity']=="ELU":
        coupling_block_nonlinearity = nn.ELU()
    elif config['coupling_block_nonlinearity']=="RELU":
        coupling_block_nonlinearity = nn.ReLU()
    else:
        raise Exception("Invalid coupling_block_nonlinearity")



    if flow_type == 'exponential_coupling':
        flow = lambda  : conditional_exponential_matrix_coupling(input_dim=flow_input_dim, context_dim=config['context_dim'], hidden_dims=hidden_dims, split_dim=None, dim=-1,device='cpu',nonlinearity=coupling_block_nonlinearity)
    elif flow_type == 'spline_coupling':
        flow = lambda : T.conditional_spline(input_dim=flow_input_dim, context_dim=config['context_dim'], hidden_dims=hidden_dims,count_bins=config["count_bins"],bound=3.0)
    elif flow_type == 'spline_autoregressive':
        flow = lambda : T.conditional_spline_autoregressive(input_dim=flow_input_dim, context_dim=config['context_dim'], hidden_dims=hidden_dims,count_bins=count_bins,bound=3)
    elif flow_type == 'affine_coupling':
        flow = lambda : T.conditional_affine_coupling(input_dim=flow_input_dim, context_dim=config['context_dim'], hidden_dims=hidden_dims)
    else:
        raise Exception(f'Invalid flow type: {flow_type}')
    if permuter_type == 'Exponential_combiner':
        permuter = lambda : Exponential_combiner(flow_input_dim)
    elif permuter_type == 'Learned_permuter':
        permuter = lambda : Learned_permuter(flow_input_dim)
    elif permuter_type == 'Full_matrix_combiner':
        permuter = lambda : Full_matrix_combiner(flow_input_dim)
    elif permuter_type == "random_permute":
        permuter = lambda : T.Permute(torch.randperm(flow_input_dim, dtype=torch.long).to(device))
    else:
        raise Exception(f'Invalid permuter type: {permuter_type}')

    conditional_flow_layers = Conditional_flow_layers(flow,config['n_flow_layers'],flow_input_dim,device,permuter,config['hidden_dims'],config['batchnorm'])

    for transform in conditional_flow_layers.transformations:
        if isinstance(transform,torch.nn.Module):
            if mode == 'train':
                transform.train()
            else:
                transform.eval()
            if data_parallel:
                transform = nn.DataParallel(transform).to(device)
            else:
                transform = transform.to(device)
            parameters+= transform.parameters()


    if config['encoder_type'] == 'pointnet2':
        encoder = Pointnet2(feature_dim=config["input_dim"]-3,out_dim=config["context_dim"])
    elif config['encoder_type'] == 'gcn':
        encoder = GCNEncoder(in_dim= config["input_dim"],out_channels=config["context_dim"],k=20)
    else:
        raise Exception('Invalid encoder type!')
    if data_parallel:
        encoder = geomDataParallel(encoder).to(device)
    else:
        encoder = encoder.to(device)
    if mode == 'train':
        encoder.train()
    else:
        encoder.eval()
    parameters += encoder.parameters()

    if config['batchnorm_encodings']:
        batchnorm_encoder = torch.nn.BatchNorm1d(config["context_dim"])
        if data_parallel:
            batchnorm_encoder = nn.DataParallel(batchnorm_encoder).to(device)
        else:
            batchnorm_encoder = batchnorm_encoder.to(device)
        parameters+= batchnorm_encoder.parameters()
        if mode == 'train':
            batchnorm_encoder.train()
        else:
            batchnorm_encoder.eval()
    else:
        batchnorm_encoder =None

    return {'parameters':parameters,"flow_layers":conditional_flow_layers,'batchnorm_encoder':batchnorm_encoder,'encoder':encoder}

def collate_double_encode(batch,input_dim):
        extract_0,extract_1 = list(zip(*batch))

        combined_data_list = [Data(x=feature_assigner(x,input_dim),pos=x[:,:3]) for x in extract_0+extract_1]
        combined_batch = Batch.from_data_list(combined_data_list)
        return combined_batch

def main(rank, world_size):

    dirs = [r'/mnt/cm-nas03/synch/students/sam/data_test/2018',r'/mnt/cm-nas03/synch/students/sam/data_test/2019',r'/mnt/cm-nas03/synch/students/sam/data_test/2020']
    #dirs = ["D:/data/cycloData/multi_scan/2018","D:/data/cycloData/multi_scan/2019","D:/data/cycloData/multi_scan/2020"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    config_path = r"config/config_conditional_encoder.yaml"
    wandb.init(project="flow_change",config = config_path)
    config = wandb.config
   
    torch.backends.cudnn.benchmark = True
    flow_input_dim = config['context_dim']

    


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

    collate = functools.partial(collate_double_encode,input_dim = config['input_dim'])
    dataloader = DataLoader(dataset,shuffle=True,batch_size=config['batch_size'],num_workers=config["num_workers"],collate_fn=collate,pin_memory=True,prefetch_factor=2,drop_last=True)


    base_dist = dist.Normal(torch.zeros(flow_input_dim).to(device), torch.ones(flow_input_dim).to(device))

    models_dict = initialize_encoder_models(config,device,mode='train')
    


    parameters = models_dict['parameters']
    encoder = models_dict['encoder']
    batchnorm_encoder = models_dict['batchnorm_encoder']
    conditional_flow_layers = models_dict['flow_layers']
    
    



    transformations = conditional_flow_layers.transformations
    
    
    flow_dist = dist.ConditionalTransformedDistribution(base_dist, transformations)
    
    if config["optimizer_type"] =='Adam':
        optimizer = torch.optim.Adam(parameters, lr=config["lr"],weight_decay=config["weight_decay"]) 
    elif config["optimizer_type"] == 'Adamax':
        optimizer = Adamax(parameters, lr=config["lr"],weight_decay=config["weight_decay"],polyak =  0.999)
    elif config["optimizer_type"] == 'AdamW':
        optimizer = torch.optim.AdamW(parameters, lr=config["lr"],weight_decay=config["weight_decay"])
    else:
        raise Exception('Invalid optimizer type!')

    

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=config["patience"],threshold=0.0001,min_lr=config["min_lr"])
    save_model_path = r'save/conditional_flow_compare'

    #Load checkpoint params if specified path
    if config['load_checkpoint']:
        print(f"Loading from checkpoint: {config['load_checkpoint']}")
        checkpoint_dict = torch.load(config['load_checkpoint'])
        conditional_flow_layers = load_transformations(checkpoint_dict,conditional_flow_layers)
        encoder.load_state_dict(checkpoint_dict['encoder'])
        if config['batchnorm_encodings']:
            batchnorm_encoder.load_state_dict(checkpoint_dict['batchnorm_encoder'])
        scheduler.load_state_dict(checkpoint_dict['scheduler'])
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    else:
        print("Starting training from scratch!")
    #Override min lr to allow for changing after checkpointing
    scheduler.min_lrs = [config['min_lr']]
    #Watch models:
    wandb.watch(encoder,idx=0)
    wandb.watch(conditional_flow_layers.transformations[0],idx=1)

    torch.autograd.set_detect_anomaly(False)



    final_model_dict = {"optimizer": optimizer,'encoder':encoder,"scheduler":scheduler.state_dict(),'batchnorm_encoder':batchnorm_encoder,'cond_distrib':flow_dist}
    for epoch in range(config["n_epochs"]):
        print(f"Starting epoch: {epoch}")
        for batch_ind,batch in enumerate(tqdm(dataloader)):
            

            optimizer.zero_grad()
            batch = batch.to(device)
            encodings = encoder(batch)
            assert not encodings.isnan().any(), "Nan in encoder"
            encoding_0, encoding_1 = torch.split(encodings,config["batch_size"])
    
            if batchnorm_encoder!=None:
                encoding_0 = batchnorm_encoder(encoding_0)
            
            
            conditioned = flow_dist.condition(encoding_0)
            
            loss = -conditioned.log_prob(encoding_1).mean()

            assert not loss.isnan(), "Nan loss!"
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters,max_norm=2.0)
            
            optimizer.step()
            
            flow_dist.clear_cache()
            
            scheduler.step(loss)
            current_lr = optimizer.param_groups[0]['lr']
            if batch_ind!=0 and  (batch_ind % int(len(dataloader)/3)  == 0):
                print(f'Making samples and saving!')
                with torch.no_grad():

                    wandb.log({'loss':loss.item(),'lr':current_lr})
                    
                    batchnorm_dict = batchnorm_encoder.state_dict() if batchnorm_encoder!=None else None
                    save_dict = {"optimizer": optimizer.state_dict(),'encoder':encoder.state_dict(),"scheduler":scheduler.state_dict(),'batchnorm_encoder':batchnorm_dict,'flow_transformations':conditional_flow_layers.make_save_list()}
                    
                    torch.save(save_dict,os.path.join(save_model_path,f"{wandb.run.name}_{epoch}_{batch_ind}_model_dict.pt"))
            else:
                wandb.log({'loss':loss.item(),'lr':current_lr})
                
            
            
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    rank=''
    main(rank,world_size)