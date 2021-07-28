from models.augmenter import Augment
import torch
from dataloaders import ChallengeDataset, AmsVoxelLoader, FullSceneLoader
import os
from utils import view_cloud_plotly,log_prob_to_color,is_valid,min_max_norm
from visualize_change_map import visualize_change
from tqdm import tqdm
import models
from torch.utils.data import DataLoader
import numpy as np
from model_initialization import inner_loop,initialize_flow,make_sample,load_flow
import matplotlib.pyplot as plt

        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inverse_map(cloud,inverse_dict):
    device = cloud.device
    return cloud*inverse_dict['furthest_distance'].to(device) + inverse_dict['mean'].to(device)



def evaluate_on_test(model_dict,config,batch_size = None,generate_samples=True):
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
            change_1_0 = log_prob_to_color(log_prob_1_0,log_prob_0_0,multiple=3.3)

            
            assert is_valid(log_prob_1_0)



            if generate_samples:
                loss, log_prob_0_1, nats = inner_loop(
                batch_0_1, model_dict, config)
            
                _,log_prob_1_1,_ = inner_loop(
                    batch_1_1, model_dict, config)
                change_0_1 = log_prob_to_color(log_prob_0_1,log_prob_1_1,multiple=3.3)

                assert is_valid(log_prob_0_1)

                sample_points_given_0 = make_sample(
                    n_points = 4000, extract_0 = voxel_0_large[0].unsqueeze(0), models_dict = model_dict, config = config,sample_distrib = None,extra_context = extra_context)
                cond_nump =  voxel_0_large[0].cpu().numpy()
                cond_nump[:, 3:6] = np.clip(
                cond_nump[:, 3:6]*255, 0, 255)
                sample_points_given_0 = sample_points_given_0.cpu().numpy().squeeze()
                sample_points_given_0[:, 3:6] = np.clip(
                sample_points_given_0[:, 3:6]*255, 0, 255)

                fig_gen_given_0 = view_cloud_plotly(sample_points_given_0[:,:3],sample_points_given_0[:,3:],show=False)
                fig_gen_given_0.write_html(f'save/examples/examples_gen/{batch_ind}_gen_given_0.html')
                fig_0 = view_cloud_plotly(voxel_0_small_original[0][:,:3],voxel_0_small_original[0][:,3:],show=False)
                fig_0.write_html(f'save/examples/examples_gen/{batch_ind}_0_small.html')
                fig_1 = view_cloud_plotly(voxel_1_small_original[0][:,:3],voxel_1_small_original[0][:,3:],show=False)
                fig_1.write_html(f'save/examples/examples_gen/{batch_ind}_1_small.html')



                sample_points_given_1 = make_sample(
                    n_points = 4000, extract_0 = voxel_opposite_large[0].unsqueeze(0), models_dict = model_dict, config = config,sample_distrib = None,extra_context = extra_context)
                sample_points_given_1 = sample_points_given_1.cpu().numpy().squeeze()
                sample_points_given_1[:, 3:6] = np.clip(
                sample_points_given_1[:, 3:6]*255, 0, 255)
                fig_gen_given_1 = view_cloud_plotly(sample_points_given_1[:,:3],sample_points_given_1[:,3:],show=False)
                fig_gen_given_1.write_html(f'save/examples/examples_gen/{batch_ind}_gen_given_1.html')

                combined_points = torch.cat((voxel_0_small_original[0][:,:3],voxel_1_small_original[0][:,:3]),dim=-1)
                combined_change = torch.cat((change_0_1[0],change_1_0[0]),dim=-1)
                combined_fig = view_cloud_plotly(combined_points,combined_change,show=False,colorscale='Bluered')
                combined_fig.write_html(f'save/examples/examples_gen/{batch_ind}_change.html')

            
            is_valid(loss)
            change_means=change_1_0.mean(dim=-1).tolist()
            change_mean_list.extend(change_means)
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
    min_non_inf = tensor[~inf_mask].min()
    tensor[inf_mask] = min_non_inf
    return tensor

def log_prob_to_color(log_prob_1_given_0, log_prob_0_given_0, multiple=3.):
    base_mean = log_prob_0_given_0.mean()
    base_std = log_prob_0_given_0.std()
    #print(f'Base  mean: {base_mean.item()}, base_std: {base_std.item()}')
     
    changed_mask_1 = (base_mean - log_prob_1_given_0) > multiple*base_std
    log_prob_1_given_0 += torch.abs(log_prob_1_given_0.min())
    log_prob_1_given_0[~changed_mask_1] = 0
    if changed_mask_1.sum()>5:
        max_change = log_prob_1_given_0[changed_mask_1].max() 
        log_prob_1_given_0[changed_mask_1] = max_change - log_prob_1_given_0[changed_mask_1] 
        log_prob_1_given_0 = min_max_norm(log_prob_1_given_0)
    else:
        log_prob_1_given_0 = torch.zeros_like(log_prob_1_given_0)
    assert is_valid(log_prob_1_given_0)
    return log_prob_1_given_0

def visualize_index(save_index,config,multiple=3.3):
    with torch.no_grad():
        dataset = FullSceneLoader(config['directory_path_train'],config['directory_path_test'], out_path='save/processed_dataset',mode='test')
        color_0_list = []
        geom_0_list = []
        change_0_list = []
        geom_1_list = []
        change_1_list = []
        color_1_list = []
        
        def get_log_prob(voxel,context_voxel,extra_context):
            voxel,context_voxel,inverse = dataset.last_processing(voxel,context_voxel)
            extra_context = extra_context if config['using_extra_context'] else None
            
            log_prob_1_given_0 = calc_change(
                [context_voxel.unsqueeze(0), voxel.unsqueeze(0),extra_context], model_dict, config).cpu()
            #is_valid(log_prob_1_given_0)
            return log_prob_1_given_0
        outputs =  dataset[save_index]              
        for data_tuple in zip(*outputs):
            voxel_0, voxel_1,voxel_0_context,voxel_1_context,extra_context = [x.cuda() for x in data_tuple]
            geom_0_list.append(voxel_0[:,:3].cpu())
            color_0_list.append(voxel_0[:,3:].cpu())
            color_1_list.append(voxel_1[:,3:].cpu())
            geom_1_list.append(voxel_1[:,:3].cpu())
            log_prob_1_given_0 = get_log_prob(voxel_1,voxel_0_context,extra_context)
            log_prob_0_given_1 = get_log_prob(voxel_0,voxel_1_context.clone(),extra_context)
            log_prob_0_given_0 = get_log_prob(voxel_0,voxel_0_context.clone(),extra_context)
            log_prob_1_given_1 = get_log_prob(voxel_1,voxel_1_context,extra_context)

            log_prob_1_given_0 = clamp_infs(log_prob_1_given_0)
            log_prob_0_given_1 = clamp_infs(log_prob_0_given_1)
            log_prob_0_given_0 = clamp_infs(log_prob_0_given_0)
            log_prob_1_given_1 = clamp_infs(log_prob_1_given_1)
            [is_valid(x) for x in [log_prob_1_given_0,log_prob_0_given_1,log_prob_0_given_0,log_prob_1_given_1]]
            change_0 = log_prob_to_color(log_prob_0_given_1,log_prob_1_given_1,multiple=multiple)
            change_1 = log_prob_to_color(log_prob_1_given_0,log_prob_0_given_0,multiple=multiple)
            change_0_list.append(change_0.cpu())
            change_1_list.append(change_1.cpu())


        geom_0 = torch.cat(geom_0_list)
        geom_1 = torch.cat(geom_1_list)
        color_0 = torch.cat(color_0_list)
        color_1 = torch.cat(color_1_list)
        change_0 = torch.cat(change_0_list)
        change_1 = torch.cat(change_1_list)
        fig_0_normal = view_cloud_plotly(
                    geom_0, color_0, show=False, title='Extract 0', point_size=5)
        fig_0_normal.write_html('0_normal.html')



        change_1[change_1!=0] = 1.
        change_0[change_0!=0] = 1.
        fig_1_normal = view_cloud_plotly(
                    geom_1, color_1, show=False, title='Extract 0', point_size=5)

        fig_1_normal.write_html('1_normal.html')
        fig_0 = view_cloud_plotly(
                    geom_0, change_0, show=False, title='Extract 0', point_size=5,colorscale='Bluered')
        fig_0.write_html('test_0.html')

        fig_1 = view_cloud_plotly(
                    geom_1, change_1, show=False, title='Extract 1', point_size=5,colorscale='Bluered')
        fig_1.write_html('test_1.html')

        pass
    

def dataset_view(dataset, index, multiple=3., gen_std=0.6, show=False,save=True,return_figs=True):
    with torch.no_grad():
        sample_distrib = models.Normal(torch.zeros(1), torch.ones(
            1)*gen_std, shape=(2000, config['latent_dim'])).to(device)
        return_dict, label = dataset[index]
        print('starting calc ')
        extract_0, extract_1 = return_dict['cloud_0'], return_dict['cloud_1']
        vars = ['gen_given_0', 'gen_given_1', 'change_1_given_0',
                'change_0_given_1', 'voxel_0', 'voxel_1']
        procesed_dict = {x: [] for x in vars}


        def get_log_prob(voxel,context_voxel):
            voxel,context_voxel,inverse = dataset.last_processing(voxel,context_voxel)
            extra_context = inverse['mean'][2].reshape(-1,1) if config['using_extra_context'] else None
            
            log_prob_1_given_0 = calc_change(
                [context_voxel.unsqueeze(0), voxel.unsqueeze(0),extra_context], model_dict, config).cpu()
            return voxel,context_voxel,inverse,log_prob_1_given_0

        def to_dev(x, device):
            try:
                return x.to(device)
            except:
                return x
        for key, val in return_dict['voxels'].items():

            context_for_1,voxel_1,context_0_0,context_for_0,voxel_0,context_1_1,z_voxel_center = [
                to_dev(x, device) for x in val]
            print(f'Vox center {z_voxel_center}')
            extra_context  = z_voxel_center.reshape((1,-1)) if config['using_extra_context'] else None



            _,context_for_1,inverse_1,log_prob_1_given_0 = get_log_prob(voxel_1,context_for_1)
            _,_,_,log_prob_0_given_0 = get_log_prob(voxel_0,context_0_0)
            _,context_for_0,inverse_0,log_prob_0_given_1 = get_log_prob(voxel_0,context_for_0)
            _,_,_,log_prob_1_given_1 = get_log_prob(voxel_1,context_1_1)

            if save:
                plt.hist(log_prob_0_given_0.numpy(),label='00')
                plt.hist(log_prob_0_given_1.numpy(),label='01')
                plt.legend(loc="upper left")
                plt.savefig(f'save/examples/{index}_{key}_hist_00and01.png')
                plt.clf()
                plt.close()


            gen_given_0 = make_sample(1024, context_for_1.unsqueeze(
                0), model_dict, config, sample_distrib=sample_distrib,extra_context=extra_context)
            gen_given_1 = make_sample(1024, context_for_0.unsqueeze(
                0), model_dict, config, sample_distrib=sample_distrib,extra_context=extra_context)
            gen_given_0[:, :3] = inverse_map(gen_given_0[:, :3],inverse_1)
            gen_given_1[:, :3] = inverse_map(gen_given_1[:, :3],inverse_0)
            procesed_dict['gen_given_0'].append(gen_given_0.cpu())
            procesed_dict['gen_given_1'].append(gen_given_1.cpu())

            procesed_dict['change_1_given_0'].append(log_prob_to_color(
                log_prob_1_given_0, log_prob_0_given_0, multiple=multiple).cpu())
            procesed_dict['change_0_given_1'].append(log_prob_to_color(
                log_prob_0_given_1, log_prob_1_given_1, multiple=multiple).cpu())

            if save:
                view_cloud_plotly(gen_given_0[:,:3],gen_given_0[:,3:],show=False).write_html(f'save/examples/{index}_{key}_gen_given_0.html')
                view_cloud_plotly(voxel_0[:,:3],voxel_0[:,3:],show=False).write_html(f'save/examples/{index}_{key}_voxel_0.html')
                view_cloud_plotly(gen_given_1[:,:3],gen_given_1[:,3:],show=False).write_html(f'save/examples/{index}_{key}_gen_given_1.html')
                view_cloud_plotly(voxel_1[:,:3],voxel_1[:,3:],show=False).write_html(f'save/examples/{index}_{key}_voxel_1.html')
          
            # voxel_0[:, :3] =  inverse_map(voxel_0[:, :3],inverse_1).cpu()
            # voxel_1[:, :3] = inverse_map(voxel_1[:, :3],inverse_0).cpu()
            procesed_dict['voxel_0'].append(voxel_0.cpu())
            procesed_dict['voxel_1'].append(voxel_1.cpu())

        procesed_dict = {key: torch.cat(val, dim=0)
                         for key, val in procesed_dict.items()}
  
        fig_gen_given_0 = view_cloud_plotly(
            procesed_dict['gen_given_0'][:, :3], procesed_dict['gen_given_0'][:, 3:], show=show, title='Gen given 0')
        fig_gen_given_1 = view_cloud_plotly(
            procesed_dict['gen_given_1'][:, :3], procesed_dict['gen_given_1'][:, 3:], show=show, title='Gen given 1')
        fig_0_given_1 = view_cloud_plotly(procesed_dict['voxel_0'][:, :3], procesed_dict['change_0_given_1'],
                                        colorscale='Bluered', show_scale=True, show=show, title='Extract 0 given 1')
        fig_1_given_0 = view_cloud_plotly(procesed_dict['voxel_1'][:, :3], procesed_dict['change_1_given_0'],
                                        colorscale='Bluered', show_scale=True, show=show, title='Extract 1 given 0')
        fig_0 = view_cloud_plotly(
            extract_0[:, :3], extract_0[:, 3:], show=show, title='Extract 0', point_size=5)
        fig_1 = view_cloud_plotly(
            extract_1[:, :3], extract_1[:, 3:], show=show, title='Extract 1', point_size=5)
        return fig_0, fig_1, fig_1_given_0, fig_0_given_1, fig_gen_given_1, fig_gen_given_0





if __name__ == '__main__':
    
    #name = load_path.split('/')[-1].split('_')[0]
    #dataset_out = f"save/processed_dataset/{name}_{mode}_probs_dataset.pt"
    #create_dataset(dataset,model_dict,dataset_out = dataset_out)
    # score_on_test(dataset,model_dict,n_bins=12)
    load_path = 'save/conditional_flow_compare/swept-energy-2784_e1_b500_model_dict.pt'
    save_dict = torch.load(load_path, map_location=device)
    config = save_dict['config']
    model_dict = initialize_flow(config, device, mode='test')
    model_dict = load_flow(save_dict, model_dict)
    mode = 'test'
    

        
    nats,log_probs_list = evaluate_on_test(model_dict,config,generate_samples=True)
    values,indices = torch.sort(torch.tensor(log_probs_list),descending=False) 
    torch.save({'values':values,'indices':indices},f'save/most_changed/{os.path.basename(load_path)}')
    one_up_path = os.path.dirname(__file__)
    out_path = os.path.join(one_up_path, r"save/processed_dataset")

    # dataset = AmsVoxelLoader(config['directory_path_train'],config['directory_path_test'], out_path='save/processed_dataset', preload=True,
    #         n_samples = config['sample_size'],final_voxel_size = config['final_voxel_size'],device=device,
    #         n_samples_context = config['n_samples_context'], context_voxel_size = config['context_voxel_size'],mode='test')


    #visualize_index(1,config,multiple=3.3)
    preload = True
    csv_path = 'save/challenge_data/Shrec_change_detection_dataset_public/new_final.csv'
    dirs = ['save/challenge_data/Shrec_change_detection_dataset_public/' +
            year for year in ["2016", "2020"]]
    dataset = ChallengeDataset(csv_path, dirs, out_path, n_samples=config['sample_size'], preload=preload, device=device, final_voxel_size=config[
                            'final_voxel_size'], n_samples_context=config['n_samples_context'], context_voxel_size=config['context_voxel_size'])
    

    
    
    #dataset_view(dataset,0,multiple = 3.,show=False)
    # # pass
    # for x in range(0,12):
    #    dataset_view(dataset,x,multiple = 3.,gen_std=0.6,save=True)
    #visualize_change(lambda index, multiple, gen_std: dataset_view(dataset, index, multiple=multiple, gen_std=gen_std,save=False), range(len(dataset)))
