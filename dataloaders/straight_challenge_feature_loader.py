import torch
import os


class StraightChallengeFeatureLoader(torch.utils.data.Dataset):
    def __init__(self,feature_dir,mode,run_name):
        paths = [os.path.join(feature_dir,x) for x in os.listdir(feature_dir) if x.split('.')[-1]=='pt' and mode in os.path.basename(x) and run_name in os.path.basename(x)]
        self.path_dict = {int(os.path.basename(x).split('_')[2]):x   for x in paths}


    def __len__(self):
        return len(self.path_dict)
    def __getitem__(self,index):
        return torch.load(self.path_dict[index])