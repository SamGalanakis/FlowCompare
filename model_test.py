from models.model_init import model_init
import torch
import torchvision
from torch import distributions
from utils import loss_fun , loss_fun_ret, view_cloud



class ModelSampler:
    def __init__(self,config_path,model_path,model_type):
        self.sample_size = sample_size
        self.model_type = model_type
        self.model_path = model_path
        self.config_path = config_path
        self.model_dict = model_init(config_path,model_path,model_type,test=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model_type == "straight":
             
            g_layers = sorted([key for key in self.model_dict.keys() if "g_block" in key])
            g_layers = [self.model_dict[key] for key in g_layers]
            prior_z = distributions.MultivariateNormal(
                torch.zeros(3), torch.eye(3)
            ) 
            
            def sample(self=self,sample_size=2000,view=True):
                with torch.no_grad():
                    x = prior_z.sample(sample_shape = (1,sample_size//5))
                    x = x.to(self.device)
                    
                    
                    for g_layer in g_layers[::-1]:
                        x = g_layer.inverse(x)
                if view:
                    view_cloud(x)
            self.sample = sample
        elif model_type == "straight_with_pointnet":
            prior_z = distributions.MultivariateNormal(
            torch.zeros(3), torch.eye(3)
            )

            prior_e = distributions.MultivariateNormal(
                torch.zeros(emb_dim), torch.eye(emb_dim)
            )
            pointnet = model_dict['pointnet']

            g_layers = sorted([key for key in model_dict.keys() if "g_block" in key])
            g_layers = [model_dict[key] for key in g_layers]

            f_layers = sorted([key for key in model_dict.keys() if "f_block" in key])
            f_layers = [model_dict[key] for key in f_layers]

            def sample(self,model_condition,sample_size=2000,view=True):
                with torch.no_grad():
                    x = prior_z.sample(sample_shape = (1,sample_size))
                    x = x.to(self.device)
                    model_condition = model_condition.to(self.device).unsqueeze(0)
                    w = pointnet(model_condition)
                    e = w
                    for g_layer in g_layers:
                        e, inter_e_ldetJ = g_layer(e)

                    for f_layer in f_layers[::-1]:
                        x = f_layer.inverse(x,e)
                if view:
                    view_cloud(x)
          
                self.sample = sample






if __name__ == '__main__':
    config_path = "config\config_straight.yaml"
    model_type = 'straight'
    model_path = "save\straight_999_fragrant-darkness-163.pt"
    sample_size = 10000
    model_sampler = ModelSampler(config_path,model_path,model_type)
    model_sampler.sample()
    
    