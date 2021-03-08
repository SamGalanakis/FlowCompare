import torch
import pyro.distributions as dist
import pyro.distributions.transforms as T
import os
import numpy as np
from utils import load_las, random_subsample,view_cloud_plotly,grid_split,extract_area,co_min_max,feature_assigner,co_standardize,sep_standardize
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
import pyro
from train_encoder_encoder_flow_compare import collate_double_encode,load_transformations,initialize_encoder_models




class Tester():
    def __init__(self,load_path,config,models_dict=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        

        
    def initialize_from_save(self,load_path):
        load_dict = torch.load(load_path)
        flow_input_dim = config['context_dim']
        base_dist = dist.Normal(torch.zeros(flow_input_dim).to(self.device), torch.ones(flow_input_dim).to(self.device))
        models_dict = initialize_encoder_models(config,self.device,mode='train')
        conditional_flow_layers = load_transformations(load_dict,models_dict['flow_layers'])
        conditional_flow_layers = conditional_flow_layers.to(self.device)

        encoder = models_dict['encoder']
        encoder.load_state_dict(load_dict['encoder'])
        self.encoder = encoder.to(self.device).eval()
        if config['batchnorm_encodings']:
            batchnorm_encoder = models_dict['batchnorm_encoder']
            batchnorm_encoder.load_state_dict(load_dict['batchnorm_encoder'])
            self.batchnorm_encoder = batchnorm_encoder.to(self.device).eval()
        transformations = conditional_flow_layers.transformations
        for transformation in transformations:
            try:
                transformation.eval()
            except:
                continue
        flow_dist = dist.ConditionalTransformedDistribution(base_dist, transformations)
        self.final_model_dict = {'encoder':encoder,'batchnorm_encoder':batchnorm_encoder,'cond_distrib':flow_dist}
    def initialize_from_model_instance(self,final_model_dict):
        
    
        self.final_model_dict = final_model_dict

    def grad_to_rgb(self,grad,dims,percentile=None):
        grad = grad[:,:dims]
        rgb = torch.sum(torch.abs(grad),axis=1).numpy()
        rgb_final = np.zeros((rgb.shape[0],3))
        if percentile!=None:
            percentile = np.percentile(rgb,95)
            rgb_final[rgb<percentile,...]= np.array([1,0,0]).astype(np.float32)
            rgb_final[rgb>percentile]= np.array([0,1,0])
        else:
            rgb_final = rgb
        return rgb_final

    def process_sample(self,cloud_0,cloud_1,make_figs=True,show=True,colorscale='Bluered',percentile=None,dims=None,save_path=False,save_name="fig"):
        if dims==None:
            dims=self.config['input_dim']
        with torch.no_grad():
            if self.config['normalization']=="min_max":
                cloud_0,cloud_1 = co_min_max(cloud_0,cloud_1)
            elif self.config['normalization']=="standardize":
                cloud_0,cloud_1 = co_standardize(cloud_0,cloud_1)
            elif self.config['normalization'] == 'sep_standardize':
                cloud_0,cloud_1 = sep_standardize(cloud_0,cloud_1)
        cloud_0 = nn.Parameter(cloud_0,requires_grad=True)
        cloud_1 = nn.Parameter(cloud_1,requires_grad=True)
        cloud_0.retain_grad()
        cloud_1.retain_grad()
        # with torch.no_grad():
        combined_batch = collate_double_encode([(cloud_0,cloud_1)],input_dim = self.config['input_dim'])
        combined_batch = combined_batch.to(device)
        encodings = self.final_model_dict['encoder'](combined_batch.to_data_list())
        assert not encodings.isnan().any(), "Nan in encoder"
        encoding_0, encoding_1 = torch.split(encodings,1)
        
        if self.config["batchnorm_encodings"]:
            encoding_0 = self.final_model_dict["batchnorm_encoder"](encoding_0)
        
        
        conditioned = self.final_model_dict['cond_distrib'].condition(encoding_0)
        
        loss = -conditioned.log_prob(encoding_1).mean()
        loss.backward()
        grads_0 = cloud_0.grad.cpu()
        rgb_0 = self.grad_to_rgb(grads_0,dims=dims,percentile=percentile)
        grads_1 = cloud_0.grad.cpu()
        rgb_1 = self.grad_to_rgb(grads_1,dims=dims,percentile=percentile)
        
        if make_figs:
            if percentile != None:
                colorscale = None
            fig_0=view_cloud_plotly(cloud_0[:,:3],rgb_0,colorscale=colorscale,show=show)
            fig_1=view_cloud_plotly(cloud_1[:,:3],rgb_1,colorscale=colorscale,show=show)
            if save_path:
                fig_0.write_html(os.path.join(save_path,f"{save_name}_0.html"))
                fig_1.write_html(os.path.join(save_path,f"{save_name}_1.html"))

        pass

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["WANDB_MODE"] = "dryrun"
    config_path = r"config\config_conditional_encoder.yaml"
    wandb.init(project="flow_change",config = config_path)
    config = wandb.config
    load_path =r"save\conditional_flow_compare\young-leaf-836_0_6864_model_dict.pt"
    tester = Tester(load_path,config)
    tester.initialize_from_save(load_path)
    input_dim =config['input_dim']
    down_sample = config['min_points']
    points_0 = load_las(r"D:\data\cycloData\2016\0_5D4KVPBP.las")[:,:input_dim]
    points_1 = load_las(r"D:\data\cycloData\2020\0_WE1NZ71I.las")[:,:input_dim]
    sign_point = np.array([86967.46,439138.8])

   
    sign_0 = extract_area(points_0,sign_point,config['grid_square_size']/2,'circle')
    sign_0 = torch.from_numpy(sign_0.astype(dtype=np.float32)).to(device)

    sign_1 = extract_area(points_1,sign_point,config['grid_square_size']/2,'circle')
    sign_1= torch.from_numpy(sign_1.astype(dtype=np.float32)).to(device)

    sign_0 = random_subsample(sign_0,down_sample)
    sign_1 = random_subsample(sign_1,down_sample)
    tester.process_sample(sign_0,sign_1,colorscale=None,dims=3,percentile=95)



    