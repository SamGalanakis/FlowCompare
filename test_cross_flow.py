import torch
from conditional_cross_flow import initialize_cross_flow,load_cross_flow,inner_loop_cross,bits_per_dim
from dataloaders import ConditionalDataGrid, ShapeNetLoader
import wandb
import os
import pyro.distributions as dist
from utils import view_cloud_plotly
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["WANDB_MODE"] = "dryrun"
config_path = r"config/config_conditional_cross.yaml"
wandb.init(project="flow_change",config = config_path)
config = wandb.config
load_path =r"save/conditional_flow_compare/polar-energy-1299_314_model_dict.pt"
save_dict = torch.load(load_path)
model_dict = initialize_cross_flow(config,device,mode='test')
model_dict = load_cross_flow(save_dict,model_dict)
mode = 'test'
if config['preselected_points']:
        scene_df_dict = {int(os.path.basename(x).split("_")[0]): pd.read_csv(os.path.join(config['dirs_challenge_csv'].replace('train',mode),x)) for x in os.listdir(config['dirs_challenge_csv'].replace('train',mode)) }
        preselected_points_dict = {key:val[['x','y']].values for key,val in scene_df_dict.items()}
        preselected_points_dict = { key:(val.unsqueeze(0) if len(val.shape)==1 else val) for key,val in preselected_points_dict.items() }
else: 
    preselected_points_dict= None
dirs = [r'/mnt/cm-nas03/synch/students/sam/data_test/2018',r'/mnt/cm-nas03/synch/students/sam/data_test/2019',r'/mnt/cm-nas03/synch/students/sam/data_test/2020']
one_up_path = os.path.dirname(__file__)
out_path = os.path.join(one_up_path,r"save/processed_dataset")
dataset=ConditionalDataGrid(dirs,out_path=out_path,preload=True,subsample=config['subsample'],sample_size=config['sample_size'],min_points=config['min_points'],grid_type='circle',normalization=config['normalization'],grid_square_size=config['grid_square_size'],preselected_points=preselected_points_dict,mode=mode)

def calc_change(extract_0,extract_1,model_dict,config,colorscale='Hot',preprocess=False):
    if preprocess:
        extract_0, extract_1 = ConditionalDataGrid.last_processing(extract_0, extract_1,config['normalization'])
    base_dist = dist.Normal(torch.zeros(config['input_dim']).to(device), torch.ones(config['input_dim']).to(device))
  
    loss,log_prob_1_given_0 = inner_loop_cross(extract_0.unsqueeze(0),extract_1.unsqueeze(0),model_dict,base_dist,config)

    return log_prob_1_given_0.squeeze()

def log_prob_to_color(log_prob_1_given_0,log_prob_0_given_0):
    changed_mask_1 = torch.abs(log_prob_1_given_0-log_prob_0_given_0.mean()) > 3*log_prob_0_given_0.std()
    log_prob_1_given_0 += torch.abs(log_prob_1_given_0.min())
    log_prob_1_given_0[~changed_mask_1] = 0
    return log_prob_1_given_0




for index in range(len(dataset)):
    extract_0, extract_1 = dataset[index]
    extract_0, extract_1 = extract_0.to(device),extract_1.to(device)
    log_prob_1_given_0 = calc_change(extract_0, extract_1,model_dict,config,preprocess=False)
    bpd = bits_per_dim(log_prob_1_given_0,6)
    log_prob_0_given_0 = calc_change(extract_0, extract_0,model_dict,config,preprocess=False)
    log_prob_0_given_1 = calc_change(extract_1, extract_0,model_dict,config,preprocess=False)
    log_prob_1_given_1 = calc_change(extract_1, extract_1,model_dict,config,preprocess=False)
    #view_cloud_plotly(extract_0[:,:3],extract_0[:,3:])
    #view_cloud_plotly(extract_1[:,:3],extract_1[:,3:])

  
    change_1_given_0 = log_prob_to_color(log_prob_1_given_0,log_prob_0_given_0)
    change_0_given_1 = log_prob_to_color(log_prob_0_given_1,log_prob_1_given_1)

    view_cloud_plotly(extract_0[:,:3],change_0_given_1,colorscale='Hot',show_scale=True)
    view_cloud_plotly(extract_1[:,:3],change_1_given_0,colorscale='Hot',show_scale=True)




