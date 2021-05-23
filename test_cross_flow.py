import torch
from train import initialize_flow,load_flow,inner_loop,make_sample
from dataloaders import ConditionalDataGrid,ChallengeDataset,AmsGridLoader
import os
from utils import view_cloud_plotly, bits_per_dim, config_loader, random_oversample
import pandas as pd
from visualize_change_map import visualize_change
import matplotlib.pyplot as plt
import  numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# config_path = r"config/100_affine_long_astral-violet-2136.yaml"
# config = config_loader(config_path)
load_path = r"save/conditional_flow_compare/sunny-waterfall-2139_e0_b33999_model_dict.pt"  # "save/conditional_flow_compare/expert-elevator-1560_12_model_dict.pt"  # r"save/conditional_flow_compare/super-pyramid-1528_372_model_dict.pt"            #r"save/conditional_flow_compare/likely-eon-1555_139_model_dict.pt"
save_dict = torch.load(load_path)
config = save_dict['config']
model_dict = initialize_flow(config,device,mode='test')
model_dict = load_flow(save_dict,model_dict)
mode = 'train'

dataset_type  = 'challenge'
one_up_path = os.path.dirname(__file__)
out_path = os.path.join(one_up_path,r"save/processed_dataset")
if dataset_type == 'challenge_grid':
    if config['preselected_points']:
            scene_df_dict = {int(os.path.basename(x).split("_")[0]): pd.read_csv(os.path.join(config['dirs_challenge_csv'].replace('train',mode),x)) for x in os.listdir(config['dirs_challenge_csv'].replace('train',mode)) }
            preselected_points_dict = {key:val[['x','y']].values for key,val in scene_df_dict.items()}
            preselected_points_dict = { key:(val.unsqueeze(0) if len(val.shape)==1 else val) for key,val in preselected_points_dict.items() }
    else: 
        preselected_points_dict= None
    dirs = [r'/mnt/cm-nas03/synch/students/sam/data_test/2018',r'/mnt/cm-nas03/synch/students/sam/data_test/2019',r'/mnt/cm-nas03/synch/students/sam/data_test/2020']
    dataset=ConditionalDataGrid(dirs,out_path=out_path,preload=True,subsample=config['subsample'],sample_size=config['sample_size'],min_points=config['min_points'],grid_type='circle',normalization=config['normalization'],grid_square_size=config['grid_square_size'],preselected_points=preselected_points_dict,mode=mode)
elif dataset_type == 'challenge':
    csv_path = 'save/challenge_data/Shrec_change_detection_dataset_public/new_final.csv'
    dirs = ['save/challenge_data/Shrec_change_detection_dataset_public/'+year for year in ["2016","2020"]]
    dataset = ChallengeDataset(csv_path, dirs, out_path,subsample="fps",sample_size=2048,preload=True,normalization=config['normalization'],radius=int(config['grid_square_size']/2),device=device)
elif dataset_type=='ams':
    dataset=AmsGridLoader('/media/raid/sam/ams_dataset/',out_path='/media/raid/sam/processed_ams',preload=config['preload'],subsample=config['subsample'],sample_size=config['sample_size'],min_points=config['min_points'],grid_type='circle',normalization=config['normalization'],grid_square_size=config['grid_square_size'])
else:
    raise Exception('invalid dataset_type')
    
def calc_change(extract_0,extract_1,model_dict,config,preprocess=False):
    if preprocess:
        extract_0, extract_1 = ConditionalDataGrid.last_processing(extract_0, extract_1,config['normalization'])
    
  
    loss,log_prob_1_given_0,_ = inner_loop(extract_0.unsqueeze(0),extract_1.unsqueeze(0),model_dict,config)

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
        save_key_index = 0 
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
            assert not log_prob_1_given_0.isnan().any()
            log_prob_0_given_0 = calc_change(extract_0, extract_0,model_dict,config,preprocess=False).to('cpu')
            log_prob_0_given_1 = calc_change(extract_1, extract_0,model_dict,config,preprocess=False).to('cpu')
            log_prob_1_given_1 = calc_change(extract_1, extract_1,model_dict,config,preprocess=False).to('cpu')
            change_0 = change_func(log_prob_1_given_1,log_prob_0_given_1)
            change_1 = change_func(log_prob_0_given_0,log_prob_1_given_0)
            save_dict[save_key_index] = {0:torch.cat((extract_0.cpu(),change_0.unsqueeze(-1).cpu()),dim=-1).cpu(),1:torch.cat((extract_1.cpu(),change_1.unsqueeze(-1).cpu()),dim=-1).cpu(),'class':label}
            save_key_index+=1
    torch.save(save_dict,dataset_out)
    print(f"Skipped {skipped}")


def dataset_view(dataset,index,multiple =3.,show=False,n_points=2000):
    
    
    extract_0, extract_1, *other = dataset[index]
   # extract_0, extract_1 = extract_0[:2000,:] , extract_1[:2000,:] # TODO Remove this !
    extract_0, extract_1 = extract_0.to(device),extract_1.to(device)
    #extract_0, extract_1 = random_oversample(extract_0,n_points),random_oversample(extract_1,n_points) #
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
    #name = load_path.split('/')[-1].split('_')[0]
    #dataset_out = f"save/processed_dataset/{name}_{mode}_probs_dataset.pt"
    #create_dataset(dataset,model_dict,dataset_out = dataset_out)
    #score_on_test(dataset,model_dict,n_bins=12)

    #dataset_view(dataset,0,multiple = 3.,show=True)
    pass
    visualize_change(lambda index,multiple: dataset_view(dataset,index,multiple = multiple,n_points = 2000),range(len(dataset)))
    




