from models.model_init import model_init
import torch
import torchvision
import numpy as np
from torch import distributions
from utils import (
    random_subsample,
    loss_fun , 
    loss_fun_ret, 
    extract_area,
    load_las,
    view_cloud_plotly
)




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
                    x = prior_z.sample(sample_shape = (1,sample_size))
                    log_probs = torch.exp(prior_z.log_prob(x.squeeze())).numpy()
                    x = x.to(self.device)
                    
                    
                    for g_layer in g_layers[::-1]:
                        x = g_layer.inverse(x)
                if view:
                    x= x.cpu().numpy().squeeze()
                    view_cloud_plotly(x,rgb= log_probs,colorscale="Hot")
            self.sample = sample
        elif model_type == "straight_with_pointnet":
            prior_z = distributions.MultivariateNormal(
            torch.zeros(3), torch.eye(3)
            )

            # prior_e = distributions.MultivariateNormal(
            #     torch.zeros(emb_dim), torch.eye(emb_dim)
            # )
            pointnet = self.model_dict['pointnet']
            #pointnet.train() #batchnorm issues?


            g_layers = sorted([key for key in self.model_dict.keys() if "g_block" in key])
            g_layers = [self.model_dict[key] for key in g_layers]

            f_layers = sorted([key for key in self.model_dict.keys() if "f_block" in key])
            f_layers = [self.model_dict[key] for key in f_layers]

            def sample(model_condition,self=self,sample_size=1000,view=True):
                with torch.no_grad():
                    x = prior_z.sample(sample_shape = (1,sample_size))
                    x = x.to(self.device)
                    
                    model_condition = model_condition.to(self.device).unsqueeze(0)
                    w = pointnet(model_condition)
                    e = w
                    for g_layer in g_layers:
                        e, inter_e_ldetJ = g_layer(e)

                    e = e.expand((x.shape[1],w.shape[0],w.shape[1])).transpose(0,1)

                    for f_layer in f_layers[::-1]:
                        x = f_layer.inverse(x,e)
                if view:
                    view_cloud(x)
          
            self.sample = sample
        else:
            raise Exception('Invalid model type!')






if __name__ == '__main__':
    config_path = "config\config_straight.yaml"
    model_type = 'straight'
    model_path = "save\straight_199_valiant-oath-238.pt_398_valiant-oath-238.pt_597_valiant-oath-238.pt_796_valiant-oath-238.pt_995_valiant-oath-238.pt"
    sample_size = 10000
    model_sampler = ModelSampler(config_path,model_path,model_type)


    # points = load_las("D:/data/cycloData/2016/0_5D4KVPBP.las")
    # sign_point = np.array([86967.46,439138.8])
    # norm_tranform = torchvision.transforms.Normalize(0,1)
    # points = extract_area(points,sign_point,1.5,'cylinder')
    # normtransf = torchvision.transforms.Lambda(lambda x: (x - x.mean(axis=0)) / x.std(axis=0))
    # norm_stand = torchvision.transforms.Lambda ( lambda x: (x - x.min(axis=0).values) / (x.max(axis=0).values - x.min(axis=0).values)     )
    # samples = [norm_stand(torch.from_numpy(random_subsample(points,1000)[:,:3])) for x in range(4)]
    # batch = torch.stack(samples).float()
    #model_sampler.sample(model_condition=samples[0].float())
    model_sampler.sample()
    
    