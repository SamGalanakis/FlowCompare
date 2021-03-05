import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import os
import numpy as np
from utils import load_las, random_subsample,view_cloud_plotly,grid_split,extract_area,co_min_max,feature_assigner,co_standardize
from torch.utils.data import Dataset,DataLoader
from itertools import permutations, combinations
from tqdm import tqdm
from models.pytorch_geometric_pointnet2 import Pointnet2
from models.nets import ConditionalDenseNN, DenseNN
from torch_geometric.data import Data,Batch
from torch_geometric.nn import fps
from dataloaders import ConditionalDataGrid, ShapeNetLoader, ConditionalVoxelGrid
import wandb
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
import torch.distributed as distributed
from models.permuters import Full_matrix_combiner,Exponential_combiner,Learned_permuter
from models.batchnorm import BatchNorm
from torch.autograd import Variable, Function
from models.Exponential_matrix_flow import conditional_exponential_matrix_coupling
from models.gcn_encoder import GCNEncoder
from models.flow_creator import Conditional_flow_layers
import torch.multiprocessing as mp
from torch_geometric.nn import DataParallel as geomDataParallel
from torch import nn
from train_encoder_encoder_flow_compare import initialize_encoder_models
import pyro
from train_encoder_encoder_flow_compare import collate_double_encode

def load_transformations(load_dict,conditional_flow_layers):
    for transformation_params,transformation in zip(load_dict['flow_transformations'],conditional_flow_layers.transformations):
        if isinstance(transformation,nn.Module):
            transformation.load_state_dict(transformation_params)
        elif isinstance(transformation,pyro.distributions.pyro.distributions.transforms.Permute):
            transformation.permutation = transformation_params
        else:
            raise Exception('How to load?')
    
    return conditional_flow_layers


class Tester():
    def __init__(self,load_path,config_path):
        load_dict = torch.load(r"save/conditional_flow_compare/1_4422_model_dict.pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        os.environ["WANDB_MODE"] = "dryrun"
        wandb.init(project="flow_change",config = config_path)
        config = wandb.config
        self.config = config
        flow_input_dim = config['context_dim']
        base_dist = dist.Normal(torch.zeros(flow_input_dim).to(self.device), torch.ones(flow_input_dim).to(self.device))

        models_dict = initialize_encoder_models(config,self.device,mode='train')
        conditional_flow_layers = load_transformations(load_dict,models_dict['flow_layers'])
        conditional_flow_layers = conditional_flow_layers.to(self.device)

        encoder = models_dict['encoder']
        encoder.load_state_dict(load_dict['encoder_dict'])
        self.encoder = encoder.to(self.device).eval()
        if config['batchnorm_encodings']:
            batchnorm_encoder = models_dict['batchnorm_encoder']
            batchnorm_encoder.load_state_dict(load_dict['batchnorm_encoder_dict'])
            self.batchnorm_encoder = batchnorm_encoder.to(self.device).eval()
        transformations = conditional_flow_layers.transformations
        for transformation in transformations:
            try:
                transformation.eval()
            except:
                continue
        self.flow_dist = dist.ConditionalTransformedDistribution(base_dist, transformations)
    def grad_to_rgb(self,grad):
        rgb = torch.sum(torch.abs(grad),axis=1).numpy()
        percentile = np.percentile(rgb,5)
        rgb_final = np.zeros((rgb.shape[0],3))
        rgb_final[rgb<percentile,...]= np.array([1,0,0]).astype(np.float32)
        rgb_final[rgb>percentile]= np.array([0,1,0])
        return rgb_final

    def process_sample(self,cloud_0,cloud_1,make_figs=True,show=True,colorscale='Bluered'):
        with torch.no_grad():
            if self.config['normalization']=="min_max":
                cloud_0,cloud_1 = co_min_max(cloud_0,cloud_1)
            elif self.config['normalization']=="standardize":
                cloud_0,cloud_1 = co_standardize(cloud_0,cloud_1)
        cloud_0 = nn.Parameter(cloud_0,requires_grad=True)
        cloud_1 = nn.Parameter(cloud_1,requires_grad=True)
        cloud_0.retain_grad()
        cloud_1.retain_grad()
        # with torch.no_grad():
        combined_batch = collate_double_encode([(cloud_0,cloud_1)],input_dim = self.config['input_dim'])
        combined_batch = combined_batch.to(device)
        encodings = self.encoder(combined_batch.to_data_list())
        assert not encodings.isnan().any(), "Nan in encoder"
        encoding_0, encoding_1 = torch.split(encodings,1)
        
        if self.config["batchnorm_encodings"]:
            encoding_0 = self.batchnorm_encoder(encoding_0)
        
        
        conditioned = self.flow_dist.condition(encoding_0)
        
        loss = conditioned.log_prob(encoding_1).mean()
        loss.backward()
        grads_0 = cloud_0.grad.cpu()
        rgb_0 = self.grad_to_rgb(grads_0)
        grads_1 = cloud_0.grad.cpu()
        rgb_1 = self.grad_to_rgb(grads_1)
        if make_figs:
            fig_0=view_cloud_plotly(cloud_0[:,:3],rgb_0,colorscale=colorscale,show=show)
            fig_1=view_cloud_plotly(cloud_1[:,:3],rgb_1,colorscale=colorscale,show=show)

        pass

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = r"config\config_conditional_encoder.yaml"
    load_path =r"save\conditional_flow_compare\1_4221_model_dict.pt"
    tester = Tester(load_path,config_path)
    input_dim =6
    down_sample = 2000
    points_0 = load_las(r"D:\data\cycloData\2016\0_5D4KVPBP.las")[:,:input_dim]
    points_1 = load_las(r"D:\data\cycloData\2020\0_WE1NZ71I.las")[:,:input_dim]
    sign_point = np.array([86967.46,439138.8])

    sign_0 = extract_area(points_0,sign_point,2,'square')
    sign_0 = torch.from_numpy(sign_0.astype(dtype=np.float32)).to(device)

    sign_1 = extract_area(points_1,sign_point,1.5,'square')
    sign_1= torch.from_numpy(sign_1.astype(dtype=np.float32)).to(device)

    sign_0 = random_subsample(sign_0,down_sample)
    sign_1 = random_subsample(sign_1,down_sample)
    tester.process_sample(sign_0,sign_1,colorscale=None)



    