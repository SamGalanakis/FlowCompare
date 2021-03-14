import torch
import os


class StraightChallengeFeatureLoader(torch.utils.data.Dataset):
    def __init__(self,feature_dir):
        paths = [os.path.join(feature_dir,feature_dir) for x in os.listdir(feature_dir) if x.split('.')[-1]=='pt']
        self.path_dict = {int(x.split('_')[0]):x   for x in paths}\


    def __len__(self):
        return len(self.path_dict)
    def __item__(self,index):
        return torch.load(self.path_dict(index))