import torch
from train import initialize_flow, load_flow, inner_loop, make_sample
from dataloaders import ChallengeDataset
import os
from utils import view_cloud_plotly
from visualize_change_map import visualize_change
from tqdm import tqdm
import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



load_path = 'save/conditional_flow_compare/clean-snowball-2546_e200_model_dict.pt'
save_dict = torch.load(load_path, map_location=device)
config = save_dict['config']
model_dict = initialize_flow(config, device, mode='test')
model_dict = load_flow(save_dict, model_dict)
mode = 'train'


one_up_path = os.path.dirname(__file__)
out_path = os.path.join(one_up_path, r"save/processed_dataset")

preload = True
csv_path = 'save/challenge_data/Shrec_change_detection_dataset_public/new_final.csv'
dirs = ['save/challenge_data/Shrec_change_detection_dataset_public/' +
        year for year in ["2016", "2020"]]
dataset = ChallengeDataset(csv_path, dirs, out_path, n_samples=config['sample_size'], preload=preload, device=device, final_voxel_size=config[
                           'final_voxel_size'], n_samples_context=config['n_samples_context'], context_voxel_size=config['context_voxel_size'])


def calc_change(extract_0, extract_1, model_dict, config):

    loss, log_prob_1_given_0, _ = inner_loop(
        extract_0.unsqueeze(0), extract_1.unsqueeze(0), model_dict, config)

    return log_prob_1_given_0.squeeze()


def log_prob_to_color(log_prob_1_given_0, log_prob_0_given_0, multiple=3.):
    changed_mask_1 = torch.abs(
        log_prob_1_given_0-log_prob_0_given_0.mean()) > multiple*log_prob_0_given_0.std()
    log_prob_1_given_0 += torch.abs(log_prob_1_given_0.min())
    log_prob_1_given_0[~changed_mask_1] = 0
    return log_prob_1_given_0


def create_dataset(dataset, model_dict, dataset_out='save/processed_dataset/'):
    def change_func(probs_same, probs_other): return torch.abs(
        probs_other-probs_same.mean())/probs_same.std()
    save_dict = {}
    skipped = 0
    with torch.no_grad():
        save_key_index = 0
        for index in tqdm(range(len(dataset))):
            extract_0, extract_1, label, _ = dataset[index]
            label_int = label.item()
            label = dataset.int_class_dict[label_int]
            extract_0, extract_1 = extract_0.to(device), extract_1.to(device)
            if extract_0.shape[0] < 20 or extract_1.shape[0] < 20:
                skipped += 1
                continue
            print(extract_0.shape, extract_1.shape)

            log_prob_1_given_0 = calc_change(
                extract_0, extract_1, model_dict, config, preprocess=False).to('cpu')
            assert not log_prob_1_given_0.isnan().any()
            log_prob_0_given_0 = calc_change(
                extract_0, extract_0, model_dict, config, preprocess=False).to('cpu')
            log_prob_0_given_1 = calc_change(
                extract_1, extract_0, model_dict, config, preprocess=False).to('cpu')
            log_prob_1_given_1 = calc_change(
                extract_1, extract_1, model_dict, config, preprocess=False).to('cpu')
            change_0 = change_func(log_prob_1_given_1, log_prob_0_given_1)
            change_1 = change_func(log_prob_0_given_0, log_prob_1_given_0)
            save_dict[save_key_index] = {0: torch.cat((extract_0.cpu(), change_0.unsqueeze(-1).cpu()), dim=-1).cpu(
            ), 1: torch.cat((extract_1.cpu(), change_1.unsqueeze(-1).cpu()), dim=-1).cpu(), 'class': label}
            save_key_index += 1
    torch.save(save_dict, dataset_out)
    print(f"Skipped {skipped}")


def dataset_view(dataset, index, multiple=3., gen_std=0.6, show=False):
    with torch.no_grad():
        sample_distrib = models.Normal(torch.zeros(1), torch.ones(
            1)*gen_std, shape=(2000, config['latent_dim'])).to(device)
        return_dict, label = dataset[index]
        print('starting calc ')
        extract_0, extract_1 = return_dict['cloud_0'], return_dict['cloud_1']
        vars = ['gen_given_0', 'gen_given_1', 'change_1_given_0',
                'change_0_given_1', 'voxel_0', 'voxel_1']
        procesed_dict = {x: [] for x in vars}

        def to_dev(x, device):
            try:
                return x.to(device)
            except:
                return x
        for key, val in return_dict['voxels'].items():

            context_0, voxel_1, context_1, voxel_0, inverse_voxel_0, inverse_voxel_1, z_voxel_center = [
                to_dev(x, device) for x in val]
            print(f'Vox center {z_voxel_center}')
            #height_adjust = torch.tensor([0.,0.,z_voxel_center.item(),0.,0.,0.]).to(device)

            log_prob_1_given_0 = calc_change(
                context_0, voxel_1, model_dict, config).cpu()
            log_prob_0_given_0 = calc_change(
                context_0, voxel_0, model_dict, config).cpu()
            log_prob_0_given_1 = calc_change(
                context_1, voxel_0, model_dict, config).cpu()
            log_prob_1_given_1 = calc_change(
                context_1, voxel_1, model_dict, config).cpu()

            gen_given_0 = make_sample(1024, context_0.unsqueeze(
                0), model_dict, config, sample_distrib=sample_distrib)
            gen_given_1 = make_sample(1024, context_1.unsqueeze(
                0), model_dict, config, sample_distrib=sample_distrib)
            gen_given_0[:, :3] = inverse_voxel_1(gen_given_0[:, :3])
            gen_given_1[:, :3] = inverse_voxel_0(gen_given_1[:, :3])
            procesed_dict['gen_given_0'].append(gen_given_0.cpu())
            procesed_dict['gen_given_1'].append(gen_given_1.cpu())

            procesed_dict['change_1_given_0'].append(log_prob_to_color(
                log_prob_1_given_0, log_prob_0_given_0, multiple=multiple).cpu())
            procesed_dict['change_0_given_1'].append(log_prob_to_color(
                log_prob_0_given_1, log_prob_1_given_1, multiple=multiple).cpu())

            voxel_0[:, :3] = inverse_voxel_0(voxel_0[:, :3]).cpu()
            voxel_1[:, :3] = inverse_voxel_1(voxel_1[:, :3]).cpu()
            procesed_dict['voxel_0'].append(voxel_0)
            procesed_dict['voxel_1'].append(voxel_1)

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

    #dataset_view(dataset,0,multiple = 3.,show=True)
    pass
    #dataset_view(dataset,0,multiple = 3.,gen_std=0.6)
    visualize_change(lambda index, multiple, gen_std: dataset_view(
        dataset, index, multiple=multiple, gen_std=gen_std), range(len(dataset)))
