from models.augmenter import Augment
import torch
from dataloaders import ChallengeDataset, AmsVoxelLoader, FullSceneLoader
import os
from utils import view_cloud_plotly,is_valid
from visualize_change_map import visualize_change
from tqdm import tqdm
import models
from torch.utils.data import DataLoader
import numpy as np
from model_initialization import inner_loop,initialize_flow,make_sample,load_flow
import pickle
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inverse_map(cloud,inverse_dict):
    device = cloud.device
    return cloud*inverse_dict['furthest_distance'].to(device) + inverse_dict['mean'].to(device)

class DatasetViewer:
    def __init__(self,model_dict,config,mode='test'):
        self.config= config
        device = 'cuda'
        
        self.model_dict = model_dict
        self.dataset = AmsVoxelLoader(config['directory_path_train'],config['directory_path_test'], out_path='save/processed_dataset', preload=True,
            n_samples = config['sample_size'],final_voxel_size = config['final_voxel_size'],device=device,
            n_samples_context = config['n_samples_context'], context_voxel_size = config['context_voxel_size'],mode=mode
             ,include_all=True)
        
    @torch.no_grad()
    def view_index(self,index, multiple=3., gen_std=0.6,hard_cutoff=None,point_size=5):
        sample_distrib = models.Normal(torch.zeros(1), torch.ones(
            1)*gen_std, shape=(2000, config['latent_dim'])).to(device)
        dataset_out = self.dataset[index]
        dataset_out = [x.to(device).unsqueeze(0) for x in dataset_out]
        voxel_0_large, voxel_1_small,extra_context, voxel_1_large_self, voxel_1_small_self, voxel_opposite_small , voxel_opposite_large,  voxel_0_small_self, voxel_0_large_self,voxel_0_small_original ,voxel_1_small_original = dataset_out
        if not self.config['using_extra_context']:
            extra_context = None
        batch_1_0 = [voxel_0_large,voxel_1_small,extra_context]
        batch_0_1 = [voxel_opposite_large,voxel_opposite_small,extra_context] 
        
        batch_0_0 = [voxel_0_large_self,voxel_0_small_self,extra_context]
        batch_1_1 = [voxel_1_large_self,voxel_1_small_self,extra_context]

        loss, log_prob_1_0, nats = inner_loop(
            batch_1_0, self.model_dict, config)
        
        _,log_prob_0_0,_ = inner_loop(
            batch_0_0, self.model_dict, config)
        assert is_valid(log_prob_1_0)
        change_1_0 = log_prob_to_change(log_prob_1_0,log_prob_0_0,multiple=multiple,hard_cutoff=hard_cutoff)
        
        assert is_valid(change_1_0)

        is_valid(loss)
        

        
        loss, log_prob_0_1, nats = inner_loop(
        batch_0_1, self.model_dict, config)
    
        _,log_prob_1_1,_ = inner_loop(
            batch_1_1, self.model_dict, config)
        change_0_1 = log_prob_to_change(log_prob_0_1,log_prob_1_1,multiple=multiple,hard_cutoff=hard_cutoff)
        
        assert is_valid(log_prob_0_1)

        sample_points_given_0 = make_sample(
            n_points = 4000, extract_0 = voxel_0_large[0].unsqueeze(0), models_dict = self.model_dict, config = config,sample_distrib = sample_distrib,extra_context = extra_context)
        cond_nump =  voxel_0_large[0].cpu().numpy()
        cond_nump[:, 3:6] = np.clip(
        cond_nump[:, 3:6]*255, 0, 255)
        sample_points_given_0 = sample_points_given_0.cpu().numpy().squeeze()
        sample_points_given_0[:, 3:6] = np.clip(
        sample_points_given_0[:, 3:6]*255, 0, 255)
        
        fig_gen_given_0 = view_cloud_plotly(sample_points_given_0[:,:3],sample_points_given_0[:,3:],show=False,point_size=point_size)

        fig_0 = view_cloud_plotly(voxel_0_small_original[0][:,:3],voxel_0_small_original[0][:,3:],show=False,point_size=point_size)

        fig_1 = view_cloud_plotly(voxel_1_small_original[0][:,:3],voxel_1_small_original[0][:,3:],show=False,point_size=point_size)




        sample_points_given_1 = make_sample(
            n_points = 4000, extract_0 = voxel_opposite_large[0].unsqueeze(0), models_dict = self.model_dict, config = config,sample_distrib = sample_distrib,extra_context = extra_context)
        sample_points_given_1 = sample_points_given_1.cpu().numpy().squeeze()
        sample_points_given_1[:, 3:6] = np.clip(
        sample_points_given_1[:, 3:6]*255, 0, 255)
        fig_gen_given_1 = view_cloud_plotly(sample_points_given_1[:,:3],sample_points_given_1[:,3:],show=False,point_size=point_size)
        

        combined_points = torch.cat((voxel_0_small_original[0][:,:3],voxel_1_small_original[0][:,:3]),dim=0)

        change_0_1[change_0_1>0] = 1.0 
        changes_0_1_count = change_0_1.sum()
        change_1_0 [change_1_0>0] = 1.0 
        changes_1_0_count = change_1_0.sum()
        combined_change = torch.cat((change_0_1,change_1_0),dim=-1).squeeze()
        
        if changes_0_1_count>0:
            fig_0_given_1 = view_cloud_plotly(voxel_0_small_original[0][:,:3],change_0_1.squeeze(),show=False,colorscale='Bluered',point_size=point_size)
        else:
            fig_0_given_1 = view_cloud_plotly(voxel_0_small_original[0][:,:3],torch.zeros_like(voxel_0_small_original[0][:,:3]) + torch.tensor([0,0,1]).to(device),show=False,point_size=point_size)
        if changes_1_0_count>0:
            fig_1_given_0 = view_cloud_plotly(voxel_1_small_original[0][:,:3],change_1_0.squeeze(),show=False,colorscale='Bluered',point_size=point_size)
        else:
            fig_1_given_0 = view_cloud_plotly(voxel_1_small_original[0][:,:3],torch.zeros_like(voxel_1_small_original[0][:,:3]) + torch.tensor([0,0,1]).to(device),show=False,point_size=point_size)

        changed_percentage = (combined_change.sum()/combined_change.numel()).item()
        print(f'Changed percentage: {changed_percentage:.2f}')
        combined_fig = view_cloud_plotly(combined_points,combined_change,show=False,colorscale='Bluered',point_size=point_size)
        
        return fig_0,fig_1,fig_gen_given_0,fig_gen_given_1,combined_fig,fig_0_given_1,fig_1_given_0,changed_percentage


    def calc_change_vals(self,path,multiple):
        if os.path.isfile(path):
            with open(path, 'rb') as fp:
                change_vals = pickle.load(fp)
                
        else:
            change_vals = []
            
            for index in tqdm(range(len(self.dataset))): 
                changed_percentage = self.view_index(index,multiple=multiple)[-1]
                change_vals.append(changed_percentage)
            with open(path, 'wb') as fp:
                pickle.dump(change_vals, fp)
        return change_vals

        
def evaluate_on_test(model_dict,config,batch_size = None,generate_samples=False):
    with torch.no_grad():
        device = 'cuda'
      
        batch_size = config['batch_size'] if batch_size == None else batch_size
        
        dataset = AmsVoxelLoader(config['directory_path_train'],config['directory_path_test'], out_path='save/processed_dataset', preload=True,
            n_samples = config['sample_size'],final_voxel_size = config['final_voxel_size'],device=device,
            n_samples_context = config['n_samples_context'], context_voxel_size = config['context_voxel_size'],mode='test',include_all=True)
        dataloader = DataLoader(dataset,batch_size=batch_size,num_workers=config['num_workers'],pin_memory=True,prefetch_factor=2, drop_last=True,shuffle=False)
        change_mean_list = []
        print(f'Evaluating on test')
        nats_avg = 0
        
        for batch_ind, batch in enumerate(tqdm(dataloader)):

            voxel_0_large, voxel_1_small,extra_context, voxel_1_large_self, voxel_1_small_self, voxel_opposite_small , voxel_opposite_large,  voxel_0_small_self, voxel_0_large_self,voxel_0_small_original ,voxel_1_small_original = [x.to(device) for x in batch]
            if not config['using_extra_context']:
                extra_context = None
            batch_1_0 = [voxel_0_large,voxel_1_small,extra_context]
            batch_0_1 = [voxel_opposite_large,voxel_opposite_small,extra_context]
            
            batch_0_0 = [voxel_0_large_self,voxel_0_small_self,extra_context]
            batch_1_1 = [voxel_1_large_self,voxel_1_small_self,extra_context]

            loss, log_prob_1_0, nats = inner_loop(
                batch_1_0, model_dict, config)
            
            _,log_prob_0_0,_ = inner_loop(
                batch_0_0, model_dict, config)
            assert is_valid(log_prob_1_0)
            change_1_0 = log_prob_to_change(log_prob_1_0,log_prob_0_0,multiple=5.4)

            
            assert is_valid(change_1_0)

            is_valid(loss)
            change_means=(change_1_0>0).float().mean(dim=-1).tolist()
            change_mean_list.extend(change_means)

            if generate_samples:
                change_val = change_means[0]
                loss, log_prob_0_1, nats = inner_loop(
                batch_0_1, model_dict, config)
            
                _,log_prob_1_1,_ = inner_loop(
                    batch_1_1, model_dict, config)
                change_0_1 = log_prob_to_change(log_prob_0_1,log_prob_1_1,multiple=5.4)

                assert is_valid(log_prob_0_1)

                sample_points_given_0 = make_sample(
                    n_points = 4000, extract_0 = voxel_0_large[0].unsqueeze(0), models_dict = model_dict, config = config,sample_distrib = None,extra_context = extra_context[0].unsqueeze(0))
                cond_nump =  voxel_0_large[0].cpu().numpy()
                cond_nump[:, 3:6] = np.clip(
                cond_nump[:, 3:6]*255, 0, 255)
                sample_points_given_0 = sample_points_given_0.cpu().numpy().squeeze()
                sample_points_given_0[:, 3:6] = np.clip(
                sample_points_given_0[:, 3:6]*255, 0, 255)
                
                fig_gen_given_0 = view_cloud_plotly(sample_points_given_0[:,:3],sample_points_given_0[:,3:],show=False)
                fig_gen_given_0.write_html(f'save/examples/test_set_changes/{change_val:.2f}_{batch_ind}_gen_given_0.html')
                fig_0 = view_cloud_plotly(voxel_0_small_original[0][:,:3],voxel_0_small_original[0][:,3:],show=False)
                fig_0.write_html(f'save/examples/test_set_changes/{change_val:.2f}_{batch_ind}_0_small.html')
                fig_1 = view_cloud_plotly(voxel_1_small_original[0][:,:3],voxel_1_small_original[0][:,3:],show=False)
                fig_1.write_html(f'save/examples/test_set_changes/{change_val:.2f}_{batch_ind}_1_small.html')



                sample_points_given_1 = make_sample(
                    n_points = 4000, extract_0 = voxel_opposite_large[0].unsqueeze(0), models_dict = model_dict, config = config,sample_distrib = None,extra_context = extra_context[0].unsqueeze(0))
                sample_points_given_1 = sample_points_given_1.cpu().numpy().squeeze()
                sample_points_given_1[:, 3:6] = np.clip(
                sample_points_given_1[:, 3:6]*255, 0, 255)
                fig_gen_given_1 = view_cloud_plotly(sample_points_given_1[:,:3],sample_points_given_1[:,3:],show=False)
                fig_gen_given_1.write_html(f'save/examples/test_set_changes/{change_val:.2f}_{batch_ind}_gen_given_1.html')

                combined_points = torch.cat((voxel_0_small_original[0][:,:3],voxel_1_small_original[0][:,:3]),dim=0)
               
                change_0_1[log_prob_0_1[0]<0] = 1.0 
                change_1_0 = torch.zeros(voxel_1_small_original.shape[1])
                change_1_0 [log_prob_1_0[0]<0] = 1.0 
                combined_change = torch.cat((change_0_1,change_1_0),dim=-1)
                combined_change[combined_change>0] = 1.0
                combined_fig = view_cloud_plotly(combined_points,combined_change,show=False,colorscale='Bluered')
                combined_fig.write_html(f'save/examples/test_set_changes/{change_val:.2f}_{batch_ind}_change.html')

            
            
            nats = nats.item()
            nats_avg = (
            nats_avg*(batch_ind) + nats)/(batch_ind+1)
        print(f'Nats: {nats_avg}')
        return nats_avg,change_mean_list



    
def calc_change(batch, model_dict, config):

    loss, log_prob_1_given_0, _ = inner_loop(batch, model_dict, config)

    return log_prob_1_given_0.squeeze()



def clamp_infs(tensor):
    inf_mask = tensor.isinf()
    if inf_mask.any():
        min_non_inf = tensor[~inf_mask].min()
        tensor[inf_mask] = min_non_inf
        print(f'Clamping infs!')
    return tensor

def log_prob_to_change(log_prob_1_given_0, log_prob_0_given_0,multiple,hard_cutoff=None):
    '''NLL to  change scaled from 0 to 1'''
    #Clamp rare -infs to min non inf val
    #print(f'Min self probs: {log_prob_0_given_0.min()}')
    
    log_prob_1_given_0 = clamp_infs(log_prob_1_given_0)
    log_prob_0_given_0 = clamp_infs(log_prob_0_given_0)
    if hard_cutoff==None:
    # Get statistics of 0 given 0 for comparison
        base_mean = log_prob_0_given_0.mean(dim=-1).unsqueeze(-1)
        base_std = log_prob_0_given_0.std(dim=-1).unsqueeze(-1)
        #print(f'Mean {base_mean.item()} std: {base_std.item()}')
        # Minimum change criterion (all values smaller than base_mean by more than multiple*base_std)
        changed_mask = log_prob_1_given_0 < base_mean - multiple*base_std
        
    
    else:
        changed_mask =  log_prob_1_given_0<hard_cutoff
    max_change = log_prob_1_given_0.max(dim=-1)[0].unsqueeze(-1)
    min_change = log_prob_1_given_0.min(dim=-1)[0].unsqueeze(-1)
    log_prob_1_given_0 = 1 - (log_prob_1_given_0 - min_change)/(max_change-min_change)

    log_prob_1_given_0[~changed_mask] = 0.0
        

    assert is_valid(log_prob_1_given_0)
    return log_prob_1_given_0




if __name__ == '__main__':
    

    load_path = 'save/conditional_flow_compare/swept-energy-2784_e1_b500_model_dict.pt'

    save_dict = torch.load(load_path, map_location=device)
    config = save_dict['config']
    model_dict = initialize_flow(config, device, mode='test')
    model_dict = load_flow(save_dict, model_dict)
    mode = 'test'
    dataset_viewer = DatasetViewer(model_dict,config,mode=mode)

    evaluate_on_test(model_dict,config,batch_size = config['batch_size'],generate_samples=False)
    #index_for_figs_list = [19857,63146,3092,152672,20579,133479,101532,182617,76605,46078,76115,49989,24434,76034]
  
    
 
    visualize_change(lambda index, multiple, gen_std, hard_cutoff,point_size: dataset_viewer.view_index(index, multiple=multiple, gen_std=gen_std,hard_cutoff=hard_cutoff,point_size=point_size), range(len(dataset_viewer.dataset)))
