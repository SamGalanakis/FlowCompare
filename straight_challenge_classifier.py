from numpy.lib.function_base import extract
import torch
import os
import numpy as np
from utils import load_las, random_subsample,view_cloud_plotly,grid_split,extract_area,co_min_max,feature_assigner,Adamax,Early_stop
from tqdm import tqdm
from dataloaders import StraightChallengeFeatureLoader,ChallengeDataset
import wandb
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch import nn
import argparse
from utils import ground_remover
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.dummy import DummyClassifier
def log_prob_to_change(log_prob_0,log_prob_1,grads_1_given_0,config,percentile=1):
  
    std_0 = log_prob_0.std()
    perc = torch.Tensor([np.percentile(log_prob_0.cpu().numpy(),percentile)]).to(log_prob_0.device)
    change = torch.zeros_like(log_prob_1)
    mask = log_prob_1<=percentile
    change[mask] = (torch.abs(log_prob_1-perc)/std_0)[mask]
    abs_grads = torch.abs(grads_1_given_0)
    grads_sum_geom = abs_grads[:,:3].sum(axis=1)
    
    grads_sum_rgb = abs_grads[:,3:].sum(axis=1)
    eps= 1e-8
    if config['input_dim']>3:
        geom_rgb_ratio = grads_sum_geom/(grads_sum_rgb+eps)
    else:
        geom_rgb_ratio = grads_sum_geom
    return change,geom_rgb_ratio

def visualize_change(dataset,feature_dataset,index,colorscale='Hot',clearance=0.5,remove_ground =True):
    extract_0 = dataset[index][0]
    extract_1 = dataset[index][1]
    feature_dict = feature_dataset[index]
   
    

   
    
    
    dataset.view(index)
    
    change, _ = log_prob_to_change(feature_dict['log_prob_0_given_0'],feature_dict['log_prob_1_given_0'],feature_dict['grads_1_given_0'],config)
    return view_cloud_plotly(extract_1[:,:3],change,colorscale = colorscale)

def summary_stats(dataset,feature_dataset,index):
    feature_dict = feature_dataset[index]
    dataset_entry  = dataset[index]
    extract_0,extract_1,label,_ = dataset_entry
    

    added = (extract_0.shape[0] ==1)
    removed = (extract_1.shape[0] ==1)
    
    if not (added or removed):
        change_given_0, geom_rgb_ratio_given_0 = log_prob_to_change(feature_dict['log_prob_0_given_0'],feature_dict['log_prob_1_given_0'],feature_dict['grads_1_given_0'],config)
        change_given_1, geom_rgb_ratio_given_1 = log_prob_to_change(feature_dict['log_prob_1_given_1'],feature_dict['log_prob_0_given_1'],feature_dict['grads_0_given_1'],config)
    else:
        print('Added' if added else 'Removed')
        return 'Added' if added else 'Removed'
    to_be_summarized = [feature_dict['log_prob_0_given_0'],feature_dict['log_prob_1_given_0'],feature_dict['log_prob_1_given_1'],feature_dict['log_prob_0_given_1'],geom_rgb_ratio_given_0,geom_rgb_ratio_given_1]
    summary_list = []
    n_changed_1 = (change_given_0!=0).sum().item()
    n_changed_0 = (change_given_1!=0).sum().item()
    for tensor in to_be_summarized:
        tensor=tensor.detach().cpu().numpy()
        statistics = list(stats.describe(tensor))
        statistics.extend(list(statistics.pop(1)))
        summary_list.extend(statistics)
    summary_list.append(n_changed_1)
    summary_list.append(n_changed_0)
    

    return summary_list,label.item()




if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    one_up_path = os.path.dirname(__file__)
    out_path = os.path.join(one_up_path,r"save/processed_dataset")

    config_path = r"config/config_straight.yaml"

    os.environ['WANDB_MODE'] = 'dryrun'
    wandb.init(project="flow_change",config = config_path)
    config = wandb.config
    dirs = [config['dir_challenge']+year for year in ["2016","2020"]]
    dataset = ChallengeDataset(config['dirs_challenge_csv'], dirs, out_path,subsample="fps",sample_size=config['sample_size'],preload=True,normalization=config['normalization'],subset=None,radius=config['radius'],remove_ground=config['remove_ground'],mode = 'train',apply_normalization=False)
    feature_dataset = StraightChallengeFeatureLoader("save/processed_dataset/straight_features/")
    X = []
    y=[]
    for index in tqdm(range(len(feature_dataset))):
        summary_out= summary_stats(dataset,feature_dataset,index)
        if isinstance(summary_out,str): continue
        X.append(summary_out[0])
        y.append(summary_out[-1])
    X=np.array(X)
    y=np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

    clf = RandomForestClassifier(random_state=0)
    dummy_clf = DummyClassifier(strategy='most_frequent')
    clf.fit(X_train, y_train)
    dummy_clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)
    dummy_pred_test = dummy_clf.predict(X_test)
    dummy_accuracy_test = accuracy_score(y_test,dummy_pred_test)
    accuracy_test = accuracy_score(y_test,y_pred_test)
    accuracy_train = accuracy_score(y_train,y_pred_train)
    # precision = precision_score(y,y_pred)
    # recall = recall_score(y,y_pred)
    # confusion = confusion_matrix(y,y_pred)
    print('Done')