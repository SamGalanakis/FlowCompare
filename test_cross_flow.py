import torch
from conditional_cross_flow import initialize_cross_flow,load_cross_flow,inner_loop_cross,bits_per_dim
from dataloaders import ConditionalDataGrid, ShapeNetLoader,ChallengeDataset
import wandb
import os
import pyro.distributions as dist
from utils import view_cloud_plotly, bin_probs
import pandas as pd
from visualize_change_map import visualize_change
import matplotlib.pyplot as plt
import  numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["WANDB_MODE"] = "dryrun"
config_path = r"config/config_conditional_cross.yaml"
wandb.init(project="flow_change",config = config_path) 
config = wandb.config
load_path = r"save/conditional_flow_compare/super-pyramid-1528_372_model_dict.pt"  # "save/conditional_flow_compare/expert-elevator-1560_12_model_dict.pt"  # r"save/conditional_flow_compare/super-pyramid-1528_372_model_dict.pt"            #r"save/conditional_flow_compare/likely-eon-1555_139_model_dict.pt"
save_dict = torch.load(load_path)
model_dict = initialize_cross_flow(config,device,mode='test')
model_dict = load_cross_flow(save_dict,model_dict)
mode = 'test'

dataset_type  = 'challenge'
one_up_path = os.path.dirname(__file__)
out_path = os.path.join(one_up_path,r"save/processed_dataset")
if dataset_type == 'multiview':
    if config['preselected_points']:
            scene_df_dict = {int(os.path.basename(x).split("_")[0]): pd.read_csv(os.path.join(config['dirs_challenge_csv'].replace('train',mode),x)) for x in os.listdir(config['dirs_challenge_csv'].replace('train',mode)) }
            preselected_points_dict = {key:val[['x','y']].values for key,val in scene_df_dict.items()}
            preselected_points_dict = { key:(val.unsqueeze(0) if len(val.shape)==1 else val) for key,val in preselected_points_dict.items() }
    else: 
        preselected_points_dict= None
    dirs = [r'/mnt/cm-nas03/synch/students/sam/data_test/2018',r'/mnt/cm-nas03/synch/students/sam/data_test/2019',r'/mnt/cm-nas03/synch/students/sam/data_test/2020']
    dataset=ConditionalDataGrid(dirs,out_path=out_path,preload=True,subsample=config['subsample'],sample_size=config['sample_size'],min_points=config['min_points'],grid_type='circle',normalization=config['normalization'],grid_square_size=config['grid_square_size'],preselected_points=preselected_points_dict,mode=mode)
elif dataset_type == 'challenge':
    dirs_challenge_csv = 'save/2016-2020-train/'.replace('train',mode)
    dirs = ['save/challenge_data/Shrec_change_detection_dataset_public/'+year for year in ["2016","2020"]]
    dataset = ChallengeDataset(dirs_challenge_csv, dirs, out_path,subsample="fps",sample_size=2000,preload=True,normalization=config['normalization'],subset=None,radius=int(config['grid_square_size']/2),remove_ground=False,mode = mode,hsv=False)
else:
    raise Exception('invalid dataset_type')
    
def calc_change(extract_0,extract_1,model_dict,config,preprocess=False):
    if preprocess:
        extract_0, extract_1 = ConditionalDataGrid.last_processing(extract_0, extract_1,config['normalization'])
    base_dist = dist.Normal(torch.zeros(config['latent_dim']).to(device), torch.ones(config['latent_dim']).to(device))
  
    loss,log_prob_1_given_0 = inner_loop_cross(extract_0.unsqueeze(0),extract_1.unsqueeze(0),model_dict,base_dist,config)

    return log_prob_1_given_0.squeeze()

def log_prob_to_color(log_prob_1_given_0,log_prob_0_given_0,multiple=3.):
    changed_mask_1 = torch.abs(log_prob_1_given_0-log_prob_0_given_0.mean()) > multiple*log_prob_0_given_0.std()
    log_prob_1_given_0 += torch.abs(log_prob_1_given_0.min())
    log_prob_1_given_0[~changed_mask_1] = 0
    return log_prob_1_given_0

def create_dataset(dataset,model_dict,dataset_out = 'save/processed_dataset/'):
    change_func = lambda probs_same,probs_other: torch.abs(probs_other-probs_same.mean())/probs_same.std()
    save_dict = {}
    skipped = 0
    with torch.no_grad():
        for index in tqdm(range(len(dataset))):
            extract_0,extract_1,label,_ = dataset[index]
            label_int = label.item()
            label = dataset.int_class_dict[label_int]
            extract_0 ,extract_1 = extract_0.to(device),extract_1.to(device)
            if extract_0.shape[0]<20 or extract_1.shape[0]<20:
                skipped +=1
                continue
            print(extract_0.shape,extract_1.shape)
            log_prob_1_given_0 = calc_change(extract_0, extract_1,model_dict,config,preprocess=False).to('cpu')
            log_prob_0_given_0 = calc_change(extract_0, extract_0,model_dict,config,preprocess=False).to('cpu')
            log_prob_0_given_1 = calc_change(extract_1, extract_0,model_dict,config,preprocess=False).to('cpu')
            log_prob_1_given_1 = calc_change(extract_1, extract_1,model_dict,config,preprocess=False).to('cpu')
            change_0 = change_func(log_prob_1_given_1,log_prob_0_given_1)
            change_1 = change_func(log_prob_0_given_0,log_prob_1_given_0)
            save_dict[index] = {0:torch.cat((extract_0.cpu(),change_0.unsqueeze(-1).cpu()),dim=-1).cpu(),1:torch.cat((extract_1.cpu(),change_1.unsqueeze(-1).cpu()),dim=-1).cpu(),'class':label}
    torch.save(save_dict,os.path.join(dataset_out,f"probs_dataset_for_postclassification.pt"))
    print(skipped)

def score_on_test(dataset,model_dict,n_bins=50,make_figs=False,dataset_out = 'save/processed_dataset/'):
    counts_dict = {}
    out_folder ='save/figs'
    combined_bins = []
    bin_labels = [f"0giv1_{index}" for index in range(n_bins)] + [f"1giv0_{index}" for index in range(n_bins)] + ['class']
    for index in tqdm(range(len(dataset))):
        extract_0,extract_1,label,_ = dataset[index]
        label_int = label.item()
        label = dataset.int_class_dict[label_int]
        extract_0 ,extract_1 = extract_0.to(device),extract_1.to(device)
        log_prob_1_given_0 = calc_change(extract_0, extract_1,model_dict,config,preprocess=False).to(device)
        log_prob_0_given_0 = calc_change(extract_0, extract_0,model_dict,config,preprocess=False).to(device)
        log_prob_0_given_1 = calc_change(extract_1, extract_0,model_dict,config,preprocess=False).to(device)
        log_prob_1_given_1 = calc_change(extract_1, extract_1,model_dict,config,preprocess=False).to(device)
        bins_1 = torch.Tensor(bin_probs(log_prob_0_given_0,log_prob_1_given_0,n_bins=n_bins))
        bins_0 = torch.Tensor(bin_probs(log_prob_1_given_1,log_prob_0_given_1,n_bins=n_bins))
        data_row = torch.cat((bins_0,bins_1)).cpu().tolist()
        data_row.append(label)
        combined_bins.append(data_row)
        if make_figs:
            if label in counts_dict:
                counts_dict[label]['bins_0'].append(bins_0)
                counts_dict[label]['bins_1'].append(bins_1)
            else:
                counts_dict[label] = {'bins_0':[],'bins_1':[]}
    if make_figs:
        figs = []
        bin_vals = np.arange(0,n_bins*0.5,0.5)
        for key,val in counts_dict.items():
            counts_0 = sum(val['bins_0'])/len(val['bins_0'])
            counts_1 = sum(val['bins_1'])/len(val['bins_1'])
            fig_0 = plt.bar(bin_vals,counts_0)
            plt.title(f'{key}, 0: counts')
            plt.xlabel('Dist from m as 0.5 multiples of std')
            plt.ylabel('Percentage')
            plt.ylim(0,1)
            plt.savefig(os.path.join(out_folder,f'{key}-0.png'))
            plt.clf()
            fig_1 = plt.bar(bin_vals,counts_0)
            plt.title(f'{key}, 1: counts')
            plt.xlabel('Dist from m as 0.5 multiples of std')
            plt.ylabel('Percentage')
            plt.ylim(0,1)
            plt.savefig(os.path.join(out_folder,f'{key}-1.png'))
            plt.clf()
    data = np.array(combined_bins)
    df = pd.DataFrame(data,columns = bin_labels)
    df_out = os.path.join(dataset_out,'test_features.csv')
    df.to_csv(df_out)
def dataset_view(dataset,index,multiple =3.,show=False):
    
    
    extract_0, extract_1, *other = dataset[index]
    extract_0, extract_1 = extract_0.to(device),extract_1.to(device)
    print('starting calc ')
    log_prob_1_given_0 = calc_change(extract_0, extract_1,model_dict,config,preprocess=False)
    bpd = bits_per_dim(log_prob_1_given_0,6)
    log_prob_0_given_0 = calc_change(extract_0, extract_0,model_dict,config,preprocess=False)
    log_prob_0_given_1 = calc_change(extract_1, extract_0,model_dict,config,preprocess=False)
    log_prob_1_given_1 = calc_change(extract_1, extract_1,model_dict,config,preprocess=False)
    fig_0 = view_cloud_plotly(extract_0[:,:3],extract_0[:,3:],show=show,title='fig_0')
    fig_1 = view_cloud_plotly(extract_1[:,:3],extract_1[:,3:],show=show,title='fig_1')

    print('loadings probs')
    change_1_given_0 = log_prob_to_color(log_prob_1_given_0,log_prob_0_given_0,multiple = multiple)
    change_0_given_1 = log_prob_to_color(log_prob_0_given_1,log_prob_1_given_1,multiple = multiple)

    fig_0_given_1 = view_cloud_plotly(extract_0[:,:3],change_0_given_1,colorscale='Bluered',show_scale=True,show=show,title='fig_0_given_1')
    fig_1_given_0 = view_cloud_plotly(extract_1[:,:3],change_1_given_0,colorscale='Bluered',show_scale=True,show=show,title='fig_1_given_0')
    return fig_0 ,fig_1,fig_1_given_0,fig_0_given_1
if __name__ == '__main__':
    create_dataset(dataset,model_dict)
    #score_on_test(dataset,model_dict,n_bins=12)
    visualize_change(lambda index,multiple: dataset_view(dataset,index,multiple = multiple),range(len(dataset)))
    




