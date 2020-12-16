from models.model_init import model_init
import torch
import torchvision
from torch import distributions
from utils import loss_fun , loss_fun_ret, view_cloud



def test_model(config_path,model_path,model_type,sample_size = 2000):

    
    model_dict = model_init(config_path,model_path,model_type,test=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "straight":
        prior_z = prior_z = distributions.MultivariateNormal(
            torch.zeros(3), torch.eye(3)
        )   
        g_layers = sorted([key for key in model_dict.keys() if "g_block" in key])
        g_layers = [model_dict[key] for key in g_layers]

        

        with torch.no_grad():
            x = prior_z.sample(sample_shape = (1,sample_size//5))
            x = x.to(device)
            
            
            for g_layer in g_layers[::-1]:
                x = g_layer.inverse(x)
            view_cloud(x)
if __name__ == '__main__':
    config_path = "config\config_straight.yaml"
    model_type = 'straight'
    model_path = "save\straight_999_fragrant-darkness-163.pt"
    sample_size = 10000
    test_model(config_path,model_path,model_type,sample_size=sample_size)
    