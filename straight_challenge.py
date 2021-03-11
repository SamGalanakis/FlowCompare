import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import load_las, random_subsample,view_cloud_plotly,grid_split,extract_area,co_min_max,feature_assigner,Adamax,Early_stop
from torch.utils.data import Dataset,DataLoader
from itertools import permutations, combinations
from tqdm import tqdm
from models.pytorch_geometric_pointnet2 import Pointnet2
from models.nets import ConditionalDenseNN, DenseNN
from torch_geometric.data import Data,Batch
from torch_geometric.nn import fps
from dataloaders import ConditionalDataGrid, ShapeNetLoader, ConditionalVoxelGrid,ChallengeDataset
import wandb
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
import torch.distributed as distributed
from models.permuters import Full_matrix_combiner,Exponential_combiner,Learned_permuter
from models.batchnorm import BatchNorm
from torch.autograd import Variable, Function
from models.Exponential_matrix_flow import exponential_matrix_coupling
from models.gcn_encoder import GCNEncoder
import torch.multiprocessing as mp
from torch_geometric.nn import DataParallel as geomDataParallel
from torch import nn
from models.flow_creator import Conditional_flow_layers
import functools
from flow_fitter import fit_flow

def load_transformations(load_dict,conditional_flow_layers):
    for transformation_params,transformation in zip(load_dict['flow_transformations'],conditional_flow_layers.transformations):
        if isinstance(transformation,nn.Module):
            transformation.load_state_dict(transformation_params)
        elif isinstance(transformation,pyro.distributions.pyro.distributions.transforms.Permute):
            transformation.permutation = transformation_params
        else:
            raise Exception('How to load?')
    
    return conditional_flow_layers



def initialize_straight_model(config,device = 'cuda',mode='train'):
    flow_input_dim = config['input_dim']
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
        flow = lambda  : exponential_matrix_coupling(input_dim=flow_input_dim, hidden_dims=hidden_dims, split_dim=None, dim=-1,device='cpu',nonlinearity=coupling_block_nonlinearity)
    elif flow_type == 'spline_coupling':
        flow = lambda : T.spline_coupling(input_dim=flow_input_dim, hidden_dims=hidden_dims,count_bins=config["count_bins"],bound=3.0)
    elif flow_type == 'spline_autoregressive':
        flow = lambda : T.spline_autoregressive(input_dim=flow_input_dim, hidden_dims=hidden_dims,count_bins=count_bins,bound=3)
    elif flow_type == 'affine_coupling':
        flow = lambda : T.affine_coupling(input_dim=flow_input_dim, hidden_dims=hidden_dims)
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
            wandb.watch(transform,log_freq=10)




    return {'parameters':parameters,"flow_layers":conditional_flow_layers}

def collate_straight(batch):
        


        return batch[0]
def train_straight_pair(parameters,transformations,config,extract_0,extract_1,device):
    
    
    extract_0,extract_1 = extract_0.to(device),extract_1.to(device)
    base_dist = dist.Normal(torch.zeros(config["input_dim"]).to(device), torch.ones(config["input_dim"]).to(device))
    flow_dist = dist.TransformedDistribution(base_dist, transformations)
    
    if config["optimizer_type"] =='Adam':
        optimizer = torch.optim.Adam(parameters, lr=config["lr"],weight_decay=config["weight_decay"]) 
    elif config["optimizer_type"] == 'Adamax':
        optimizer = Adamax(parameters, lr=config["lr"],weight_decay=config["weight_decay"],polyak =  0.999)
    elif config["optimizer_type"] == 'AdamW':
        optimizer = torch.optim.AdamW(parameters, lr=config["lr"],weight_decay=config["weight_decay"])
    elif config["optimizer_type"] == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=config["lr"], momentum=0.9)
    else:
        raise Exception('Invalid optimizer type!')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=config["patience"],threshold=0.0001,min_lr=config["min_lr"])
    
    early_stopper = Early_stop(patience=config["patience_stopper"],min_perc_improvement=config['early_stop_margin'])
    for epoch in range(config["n_epochs"]):
        optimizer.zero_grad()
        input_data = extract_0.clone()
        input_data += torch.randn_like(input_data).to(device)*(0.01)
        assert not input_data.isnan().any()
        loss = -flow_dist.log_prob(input_data.squeeze()).mean()
        assert not loss.isnan()
        
        loss.backward()
        optimizer.step()
        
 
        scheduler.step(loss)
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({'loss': loss.item(),"lr" : current_lr})
        print(loss)
        stop_train = early_stopper.log(loss.cpu())
        flow_dist.clear_cache()
        if stop_train:
            print(f"Early stopped at epoch: {epoch}!")
            break
    for transformation in transformations:
        try:
            transformation = transformation.eval()
        except:
            continue
    
    extract_1 = nn.Parameter(extract_1,requires_grad=True)
    extract_1.retain_grad()
    with torch.no_grad():
        log_prob_0 = flow_dist.log_prob(extract_0).squeeze()
    log_prob_1 = flow_dist.log_prob(extract_1).squeeze()
    log_prob_1.mean().backward()
    grads_1 = extract_1.grad.squeeze()
    return log_prob_0,log_prob_1,grads_1

def log_prob_to_change(log_prob_0,log_prob_1,grads_1_given_0,config,percentile=1):
    std_0 = log_prob_0.std()
    perc = torch.Tensor([np.percentile(log_prob_0.cpu().numpy(),percentile)]).cuda()
    change = torch.zeros_like(log_prob_1)
    mask = log_prob_1<=percentile
    change[mask] = torch.abs(log_prob_1-perc)/std_0
    relevant_grads = torch.abs(grads_1_given_0[mask,...])
    grads_sum_geom = relevant_grads[:,:3].sum(axis=0)
    
    grads_sum_rgb = relevant_grads[:,3:].sum(axis=0)
    eps= 1e-8
    if config['input_dim']>3:
        geom_rgb_ratio = grads_sum_geom/(grads_sum_rgb+eps)
    else:
        geom_rgb_ratio = grads_sum_geom

    return change,geom_rgb_ratio
def main(rank, world_size):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    one_up_path = os.path.dirname(__file__)
    out_path = os.path.join(one_up_path,r"save/processed_dataset")
    
    config_path = r"config\config_straight.yaml"
    wandb.init(project="flow_change",config = config_path)
    config = wandb.config
    
    torch.backends.cudnn.benchmark = True
 
    
    


    one_up_path = os.path.dirname(__file__)
    out_path = os.path.join(one_up_path,r"save/processed_dataset")
    dirs = [config['dir_challenge']+year for year in ["2016","2020"]]
    if config['data_loader'] == 'ConditionalDataGridSquare':
        dataset=ConditionalDataGrid(config['dirs_challenge'],out_path=out_path,preload=config['preload'],subsample=config["subsample"],sample_size=config["sample_size"],min_points=config["min_points"],grid_type='square',normalization=config['normalization'],grid_square_size=config['grid_square_size'])
    elif config['data_loader'] == 'ConditionalDataGridCircle':
        dataset=ConditionalDataGrid(config['dirs_challenge'],out_path=out_path,preload=config['preload'],subsample=config['subsample'],sample_size=config['sample_size'],min_points=config['min_points'],grid_type='circle',normalization=config['normalization'],grid_square_size=config['grid_square_size'])
    elif config['data_loader']=='ShapeNet':
        dataset = ShapeNetLoader(r'D:\data\ShapeNetCore.v2.PC15k\02691156\train',out_path=out_path,preload=config['preload'],subsample=config['subsample'],sample_size=config['sample_size'])
    elif config['data_loader']=='ChallengeDataset':
        dataset = ChallengeDataset(config['dirs_challenge_csv'], dirs, out_path,subsample="fps",sample_size=config['sample_size'],preload=config['preload'],normalization=config['normalization'])
    else:
        raise Exception('Invalid dataloader type!')


    dataloader = DataLoader(dataset,shuffle=False,batch_size=config['batch_size'],num_workers=config["num_workers"],collate_fn=collate_straight,pin_memory=True,prefetch_factor=2,drop_last=False)



    
    torch.autograd.set_detect_anomaly(False)

    for index, batch in enumerate(tqdm(dataloader)):
        batch = [x.to(device) for x in batch]
        extract_0, extract_1, label = batch
        #fit_flow(extract_0,extract_1)
        extract_0 , extract_1 = extract_0[:,:config['input_dim']].unsqueeze(0),extract_1[:,:config['input_dim']].unsqueeze(0)
        #Initialize models
        models_dict = initialize_straight_model(config,device,mode='train')
        parameters = models_dict['parameters']
        conditional_flow_layers = models_dict['flow_layers']
        transformations = conditional_flow_layers.transformations
        log_prob_0_given_0,log_prob_1_given_0,grads_1_given_0 = train_straight_pair(parameters,transformations,config,extract_0,extract_1,device=device)
        change_1,geom_rgb_ratio_1 = log_prob_to_change(log_prob_0_given_0,log_prob_1_given_0,grads_1_given_0,config=config,percentile=1)
        #Reinitialize models
        models_dict = initialize_straight_model(config,device,mode='train')
        parameters = models_dict['parameters']
        conditional_flow_layers = models_dict['flow_layers']
        transformations = conditional_flow_layers.transformations
        log_prob_1_given_1,log_prob_0_given_1,grads_0_given_1 = train_straight_pair(parameters,transformations,config,extract_1,extract_0,device=device)
        change_0,geom_rgb_ratio_0 = log_prob_to_change(log_prob_1_given_1,log_prob_0_given_1,grads_0_given_1,config=config,percentile=1)
        #Get change
                
            
            
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    rank=''
    main(rank,world_size)