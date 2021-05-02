import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import os
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch.distributions.utils import _sum_rightmost
from tqdm import tqdm
from dataloaders import ConditionalDataGrid, ShapeNetLoader,AmsGridLoader
import wandb
from torch import nn
import pandas as pd
import einops
import math
from time import time,perf_counter
from models import (
ExponentialCombiner,
NeighborhoodEmbedder,
get_cross_attn,
ExponentialCoupling,
DGCNNembedder,
GCNembedder,
DGCNNembedderCombo,
MLP,
Permuter,
Augment,
StandardUniform,
StandardNormal,
get_cif_block_attn,
ConditionalMeanStdNormal,
Flow,
IdentityTransform
)


def load_cross_flow(load_dict,initialized_cross_flow):
    
    initialized_cross_flow['augmenter'].load_state_dict(load_dict['augmenter'])
    initialized_cross_flow['base_dist'].load_state_dict(load_dict['base_dist'])
    initialized_cross_flow['input_embedder'].load_state_dict(load_dict['input_embedder'])

    for layer_dicts,layer in zip(load_dict['layers'],initialized_cross_flow['layers']):
        for key,val in layer.items():
                if isinstance(val,nn.Module):
                    val.load_state_dict(layer_dicts[key])
                else:
                    raise Exception('How to load?')
    return initialized_cross_flow



def initialize_cross_flow(config,device = 'cuda',mode='train'):
    

    parameters = []
    
    if config['coupling_block_nonlinearity']=="ELU":
        coupling_block_nonlinearity = nn.ELU()
    elif config['coupling_block_nonlinearity']=="RELU":
        coupling_block_nonlinearity = nn.ReLU()
    else:
        raise Exception("Invalid coupling_block_nonlinearity")

    if config['latent_dim'] > config['input_dim']:

        if config['augmenter_dist'] == 'StandardUniform':
            augmenter_dist = StandardUniform(shape = (config['sample_size'],config['latent_dim']-config['input_dim']))
        elif config['augmenter_dist'] == 'ConditionalMeanStdNormal':
            net_augmenter_dist = MLP(config['input_dim'],config['net_augmenter_dist_hidden_dims'],config['latent_dim']-config['input_dim'],coupling_block_nonlinearity)
            augmenter_dist = ConditionalMeanStdNormal( net = net_augmenter_dist,scale_shape =  config['latent_dim']-config['input_dim'])
        else: 
            raise Exception('Invalid augmenter_dist')
    

        augmenter = Augment(augmenter_dist,split_dim=-1,x_size = config['input_dim'])
    else:
        augmenter = IdentityTransform().to(device)

 
  

    
    

    if config['flow_type'] == 'affine_coupling':
        flow = lambda : affine_coupling_attn(config['input_dim'],config['attn_dim'],hidden_dims= config['hidden_dims'])
    elif config['flow_type'] == 'exponential_coupling':
        flow_for_cif = lambda input_dim,context_dim: ExponentialCoupling(input_dim,context_dim = context_dim,nonlinearity = coupling_block_nonlinearity,hidden_dims= config['hidden_dims'], eps_expm = config['eps_expm']) 
        #flow_with_attn = lambda : ExponentialCoupling(input_dim=config['latent_dim'],context_dim = config['attn_dim'],nonlinearity = coupling_block_nonlinearity,hidden_dims= config['hidden_dims'], eps_expm = config['eps_expm'])
        #plain_flow = lambda : ExponentialCoupling(input_dim=config['latent_dim'],context_dim = None,nonlinearity = coupling_block_nonlinearity,hidden_dims= config['hidden_dims'], eps_expm = config['eps_expm'])
        
    else:
        raise Exception('Invalid flow type')
    
    
    #out_dim,query_dim, context_dim, heads, dim_head, dropout
    attn = lambda : get_cross_attn(config['attn_dim'],config['attn_input_dim'],config['input_embedding_dim'],config['cross_heads'],config['cross_dim_head'],config['attn_dropout'])


    

    pre_attention_mlp = lambda input_dim_pre_attention_mlp: MLP(input_dim_pre_attention_mlp,config['pre_attention_mlp_hidden_dims'],config['attn_input_dim'],coupling_block_nonlinearity,residual=True)
    
    
    if config['permuter_type'] == 'Exponential_combiner':
        permuter = lambda dim: ExponentialCombiner(dim)
    elif config['permuter_type'] == "random_permute":
        permuter = lambda dim: Permuter(permutation = torch.randperm(dim, dtype=torch.long).to(device))
    else:
        raise Exception(f'Invalid permuter type: {config["""permuter_type"""]}')



    if config['cif_dist'] == 'StandardUniform':
        cif_dist = lambda : StandardUniform(shape = (config['sample_size'],config['cif_latent_dim']-config['latent_dim']))
    elif config['cif_dist'] == 'ConditionalMeanStdNormal':
        net_cif_dist = MLP(config['input_dim'],config['net_cif_dist_hidden_dims'],config['augmenter_dim']-config['latent_dim'],coupling_block_nonlinearity)
        cif_dist = lambda : ConditionalMeanStdNormal( net = net_cif_dist,scale_shape =  config['augmenter_dim']-config['latent_dim'])
    else: 
        raise Exception('Invalid cif_dist')
     
    cif_block = lambda : get_cif_block_attn(config['latent_dim'],config['cif_latent_dim'],cif_dist,config['attn_dim'],flow_for_cif,attn,pre_attention_mlp,config['n_flows_cif'],event_dim=-1,permuter=permuter)
   
    

    transforms = []
    transforms.append(augmenter)
    #Add transformations to list
    for index in range(config['n_flow_layers']):
        transforms.append(cif_block())
        #Don't permute output
        if index != config['n_flow_layers']-1:
            transforms.append(permuter(config['latent_dim']))

    final_flow = Flow(transforms)
      
    
   


    if config['input_embedder'] == 'NeighborhoodEmbedder':
        input_embedder = NeighborhoodEmbedder(config['input_dim'],out_dim = config['input_embedding_dim'])
    elif config['input_embedder'] == 'DGCNNembedder':
        input_embedder = DGCNNembedder(emb_dim= config['input_embedding_dim'],n_neighbors=config['n_neighbors'])
    elif config['input_embedder'] == 'DGCNNembedderCombo':
        input_embedder = DGCNNembedderCombo(config['input_embedding_dim'],config['global_input_embedding_dim'],n_neighbors=config['n_neighbors'])

    elif config['input_embedder'] == 'idenity':
        input_embedder = nn.Identity()
    else:
        raise Exception('Invalid input embeder!')

    if mode == 'train':
        input_embedder.train()
        final_flow.train()
     
    else:
        input_embedder.eval()
        final_flow.eval()
        
    if config['data_parallel']:
        input_embedder = nn.DataParallel(input_embedder).to(device)
        final_flow = nn.DataParallel(final_flow).to(device)

    else:
        input_embedder = input_embedder.to(device)
        final_flow = final_flow.to(device)

    
  
    parameters += input_embedder.parameters()
    parameters += final_flow.parameters()

    models_dict = {'parameters':parameters,"flow":final_flow,'input_embedder':input_embedder}
    

    


    base_dist =  StandardNormal(shape = (config['sample_size'],config['latent_dim']-config['input_dim'])).to(device)
    parameters += base_dist.parameters()
    
    models_dict['base_dist'] = base_dist
    print(f'Number of trainable parameters: {sum([x.numel() for x in parameters])}')
    return models_dict


def inner_loop_cross(extract_0,extract_1,models_dict,config):
    

    input_embeddings = models_dict["input_embedder"](extract_0)
    
    x= extract_1
    
    x,ldj = models_dict['flow'](x,context = input_embeddings)
    

    log_prob = models_dict["base_dist"].log_prob(x) + ldj
    loss = -log_prob.sum() / (math.log(2) * x.numel())
    
    return loss,log_prob
def sample_cross(n_samples,extract_0,models_dict,base_dist_for_sample,config):

    input_embeddings = models_dict["input_embedder"](extract_0)

    x = base_dist_for_sample.sample(config['batch_size'])
    
    x = models_dict['flow'].inverse(x,context=input_embeddings)
    return x


def main(rank, world_size):

    dirs = [r'/mnt/cm-nas03/synch/students/sam/data_test/2018',r'/mnt/cm-nas03/synch/students/sam/data_test/2019',r'/mnt/cm-nas03/synch/students/sam/data_test/2020']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    

    config_path = r"config/config_conditional_cross.yaml"
    wandb.init(project="flow_change",config = config_path)
    config = wandb.config

    models_dict = initialize_cross_flow(config,device,mode='train')
   

    if config['preselected_points']:
        scene_df_dict = {int(os.path.basename(x).split("_")[0]): pd.read_csv(os.path.join(config['dirs_challenge_csv'],x)) for x in os.listdir(config['dirs_challenge_csv']) }
        preselected_points_dict = {key:val[['x','y']].values for key,val in scene_df_dict.items()}
        preselected_points_dict = { key:(val.unsqueeze(0) if len(val.shape)==1 else val) for key,val in preselected_points_dict.items() }
    else: 
        preselected_points_dict= None

    one_up_path = os.path.dirname(__file__)
    out_path = os.path.join(one_up_path,r"save/processed_dataset")
    if config['data_loader'] == 'ConditionalDataGridSquare':
        dataset=ConditionalDataGrid(dirs,out_path=out_path,preload=config['preload'],subsample=config["subsample"],sample_size=config["sample_size"],min_points=config["min_points"],grid_type='square',normalization=config['normalization'],grid_square_size=config['grid_square_size'],preselected_points=preselected_points_dict)
    elif config['data_loader'] == 'ConditionalDataGridCircle':
        dataset=ConditionalDataGrid(dirs,out_path=out_path,preload=config['preload'],subsample=config['subsample'],sample_size=config['sample_size'],min_points=config['min_points'],grid_type='circle',normalization=config['normalization'],grid_square_size=config['grid_square_size'],preselected_points=preselected_points_dict)
    elif config['data_loader']=='ShapeNet':
        dataset = ShapeNetLoader(r'D:\data\ShapeNetCore.v2.PC15k\02691156\train',out_path=out_path,preload=config['preload'],subsample=config['subsample'],sample_size=config['sample_size'])
    elif config['data_loader'] == 'AmsGridLoader':
        dataset=AmsGridLoader('/media/raid/sam/ams_dataset/',out_path='/media/raid/sam/processed_ams',preload=config['preload'],subsample=config['subsample'],sample_size=config['sample_size'],min_points=config['min_points'],grid_type='circle',normalization=config['normalization'],grid_square_size=config['grid_square_size'])

    else:
        raise Exception('Invalid dataloader type!')

  
    dataloader = DataLoader(dataset,shuffle=True,batch_size=config['batch_size'],num_workers=config["num_workers"],collate_fn=None,pin_memory=True,prefetch_factor=2,drop_last=True)


    
    

    
    



    if config["optimizer_type"] =='Adam':
        optimizer = torch.optim.Adam(models_dict['parameters'], lr=config["lr"],weight_decay=config["weight_decay"]) 
    elif config["optimizer_type"] == 'Adamax':
        optimizer = torch.optim.Adamax(models_dict['parameters'], lr=config["lr"],weight_decay=config["weight_decay"],polyak =  0.999)
    elif config["optimizer_type"] == 'AdamW':
        optimizer = torch.optim.AdamW(models_dict['parameters'], lr=config["lr"],weight_decay=config["weight_decay"])
    else:
        raise Exception('Invalid optimizer type!')

    
    if config['lr_scheduler'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=config["patience"],threshold=0.0001,min_lr=config["min_lr"])
    elif config['lr_scheduler'] == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = config['lr'], epochs=config['n_epochs'], steps_per_epoch=len(dataloader),total_steps =len(dataloader))
    else:
        raise Exception('Invalid cheduler')
    
    save_model_path = r'save/conditional_flow_compare'

    #Load checkpoint params if specified path
    if config['load_checkpoint']:
        print(f"Loading from checkpoint: {config['load_checkpoint']}")
        checkpoint_dict = torch.load(config['load_checkpoint'])
        models_dict = load_cross_flow(checkpoint_dict,models_dict)
        scheduler.load_state_dict(checkpoint_dict['scheduler'])
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    else:
        print("Starting training from scratch!")
    #Override min lr to allow for changing after checkpointing
    scheduler.min_lrs = [config['min_lr']]
    #Watch models:


    torch.autograd.set_detect_anomaly(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    
    scaler = torch.cuda.amp.GradScaler(enabled=config['amp'])

    for epoch in range(config["n_epochs"]):
        print(f"Starting epoch: {epoch}")
        loss_running_avg = 0
        for batch_ind,batch in enumerate(tqdm(dataloader)):
            with torch.cuda.amp.autocast(enabled=config['amp']):
                torch.cuda.synchronize()
                t0 = perf_counter()
                batch = [x.to(device) for x in batch]
                extract_0,extract_1 = batch
    

                loss, _ = inner_loop_cross(extract_0,extract_1,models_dict,config)
    
            
            scaler.scale(loss).backward()
      


            torch.nn.utils.clip_grad_norm_(models_dict['parameters'],max_norm=config['grad_clip_val'])
            scaler.step(optimizer)
            scaler.update()

            
            optimizer.zero_grad(set_to_none=True)
            current_lr = optimizer.param_groups[0]['lr']
            torch.cuda.synchronize()
            time_batch = perf_counter() - t0
            loss_item = loss.item()
            loss_running_avg = (loss_running_avg*(batch_ind) + loss_item)/(batch_ind+1)
            
            if (batch_ind+1)%config['batches_per_sample'] == 0:
                if not config['attn_connection']:
                    with torch.no_grad():
                        #Multiply std to get tighter samples
                        base_dist_for_sample = torch.distributions.Normal(torch.zeros(config['latent_dim']).to(device), torch.ones(config['latent_dim']).to(device)*0.6)
                        sample = sample_cross(4000,extract_0,models_dict,base_dist_for_sample,config)[0]
                        sample = sample.cpu().numpy().squeeze()
                        sample[:,3:6] = np.clip(sample[:,3:6]*255,0,255)
                        cond_nump = extract_0.cpu().numpy()[0]
                        cond_nump[:,3:6] = np.clip(cond_nump[:,3:6]*255,0,255)
                        wandb.log({"Cond_cloud": wandb.Object3D(cond_nump[:,:6]),"Gen_cloud": wandb.Object3D(sample[:,:6])})
            else:
                wandb.log({'loss':loss_item,'lr':current_lr,'time_batch':time_batch})
            
        scheduler.step(loss_running_avg)
        save_dict = {"optimizer": optimizer.state_dict(),"scheduler":scheduler.state_dict(),"flow":models_dict['flow'].state_dict(),"input_embedder":models_dict['input_embedder'].state_dict()}
        torch.save(save_dict,os.path.join(save_model_path,f"{wandb.run.name}_{epoch}_model_dict.pt"))
        wandb.log({'epoch':epoch,"loss_epoch":loss_running_avg})
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    rank=''
    main(rank,world_size)