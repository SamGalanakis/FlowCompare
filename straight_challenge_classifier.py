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
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.dummy import DummyClassifier
import pickle
from sklearn.linear_model import LogisticRegression











def bin_probs(log_probs_0,log_probs_1,n_bins=50):
    
    std0 = log_probs_0.std()
    m0 = log_probs_0.mean()
    bin_width = 0.5*std0
    bin_counts = [0]*50
    bins,counts = torch.unique(torch.abs(log_probs_1-m0)//bin_width,return_counts=True)
    normalized_counts = counts/log_probs_1.shape[0]
    for bin,normalized_count in list(zip(bins,normalized_counts)):
        if int(bin)>=n_bins:
            break
        bin_counts[int(bin)] = normalized_count.item()
        

    

    return bin_counts






def log_prob_to_change(log_prob_0,log_prob_1,full_probs_1_given_0,config,percentile=1):
    full_probs_1_given_0 = full_probs_1_given_0.squeeze()
    std_0 = log_prob_0.std()
    perc = torch.Tensor([np.percentile(log_prob_0.cpu().numpy(),percentile)]).to(log_prob_0.device)
    change = torch.zeros_like(log_prob_1)
    mask = log_prob_1<=percentile
    change[mask] = (torch.abs(log_prob_1-perc)/std_0)[mask]
    
    full_probs_sum_geom = full_probs_1_given_0[:,:3].sum(axis=1)
    
    full_probs_sum_rgb = full_probs_1_given_0[:,3:].sum(axis=1)
    eps= 1e-8
    if config['input_dim']>3:
        geom_rgb_ratio = full_probs_sum_geom/(full_probs_sum_rgb+eps)
    else:
        geom_rgb_ratio = full_probs_sum_geom
    return change,geom_rgb_ratio

def visualize_change(dataset,feature_dataset,index,colorscale='Hot'):
    extract_0 = dataset[index][0]
    extract_1 = dataset[index][1]
    feature_dict = feature_dataset[index]
    added = (extract_0.shape[0] ==1)
    removed = (extract_1.shape[0] ==1)
    
    if not (added or removed):
        change_given_0, geom_rgb_ratio_given_0 = log_prob_to_change(feature_dict['log_prob_0_given_0'],feature_dict['log_prob_1_given_0'],feature_dict['full_probs_1_given_0'],config)
        change_given_1, geom_rgb_ratio_given_1 = log_prob_to_change(feature_dict['log_prob_1_given_1'],feature_dict['log_prob_0_given_1'],feature_dict['full_probs_0_given_1'],config)
    else:
        print('added' if added else 'removed')
    

   
    
    
    dataset.view(index)
    
    
    rgb_change = torch.zeros_like(change_given_0)
    rgb_change[change_given_0!=0] =1
    view_cloud_plotly(extract_1[:,:3],rgb_change,colorscale = colorscale)
    view_cloud_plotly(extract_1[:,:3],-feature_dict['log_prob_1_given_0'],colorscale = colorscale)
    return view_cloud_plotly(extract_1[:,:3],change_given_0,colorscale = colorscale)

def summary_stats(dataset,feature_dataset,index):
    feature_dict = feature_dataset[index]
    dataset_entry  = dataset[index]
    extract_0,extract_1,label,_ = dataset_entry
    

    added = (extract_0.shape[0] ==1)
    removed = (extract_1.shape[0] ==1)
    
    if not (added or removed):
        bin_probs_1 = bin_probs(feature_dict['log_prob_0_given_0'],feature_dict['log_prob_1_given_0'],n_bins=50)
        bin_probs_0 = bin_probs(feature_dict['log_prob_1_given_1'],feature_dict['log_prob_0_given_1'],n_bins=50)
        change_given_0, geom_rgb_ratio_given_0 = log_prob_to_change(feature_dict['log_prob_0_given_0'],feature_dict['log_prob_1_given_0'],feature_dict['full_probs_1_given_0'],config)
        change_given_1, geom_rgb_ratio_given_1 = log_prob_to_change(feature_dict['log_prob_1_given_1'],feature_dict['log_prob_0_given_1'],feature_dict['full_probs_0_given_1'],config)
    else:
        print('added' if added else 'removed')
        return dataset.class_int_dict['added' if added else 'removed'],label.item()
    to_be_summarized = [feature_dict['log_prob_0_given_0'],feature_dict['log_prob_1_given_0'],feature_dict['log_prob_1_given_1'],feature_dict['log_prob_0_given_1'],geom_rgb_ratio_given_0,geom_rgb_ratio_given_1]
    summary_list = []

    summary_list.extend([geom_rgb_ratio_given_0.mean().item(),geom_rgb_ratio_given_1.mean().item()])
    summary_list.extend(bin_probs_1)
    summary_list.extend(bin_probs_0)
    
    return summary_list,label.item()




if __name__ == '__main__':
    preload_classification_data= False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    one_up_path = os.path.dirname(__file__)
    out_path = os.path.join(one_up_path,r"save/processed_dataset")

    config_path = r"config/config_straight.yaml"

    os.environ['WANDB_MODE'] = 'dryrun'
    wandb.init(project="flow_change",config = config_path)
    config = wandb.config
    dirs = [config['dir_challenge']+year for year in ["2016","2020"]]
    dataset_test = ChallengeDataset("save/2016-2020-test/", dirs, out_path,subsample="fps",sample_size=config['sample_size'],preload=True,normalization=config['normalization'],subset=None,radius=config['radius'],remove_ground=config['remove_ground'],mode = 'test',apply_normalization=False)
    feature_dataset_test = StraightChallengeFeatureLoader("save/processed_dataset/straight_features/",'test',run_name = '1500epochs')

    feature_dataset_train = StraightChallengeFeatureLoader("save/processed_dataset/straight_features/",'train',run_name = '1500epochs')
    
    
    dataset_train = ChallengeDataset("save/2016-2020-train/", dirs, out_path,subsample="fps",sample_size=config['sample_size'],preload=True,normalization=config['normalization'],subset=None,radius=config['radius'],remove_ground=config['remove_ground'],mode = 'train',apply_normalization=False)
   
    def process_features(dataset,feature_dataset):
        X = []
        y=[]
        y_preclassified =[]
        y_preclassified_predicted =[]
        for index in tqdm(range(len(feature_dataset))):
            summary_out= summary_stats(dataset,feature_dataset,index)
            if isinstance(summary_out[0],int): 
                y_preclassified.append(summary_out[-1])
                y_preclassified_predicted.append(summary_out[0])
                continue
            X.append(summary_out[0])
            y.append(summary_out[-1])
    
        return np.array(X),np.array(y),np.array(y_preclassified),np.array(y_preclassified_predicted)


    save_path  = 'save/challenge_classifier_save.pickle'

    if not preload_classification_data:
        X_test,y_test,y_preclassified_test,y_preclassified_predicted_test = process_features(dataset_test,feature_dataset_test)
        X_train,y_train,y_preclassified_train,y_preclassified_predicted_train = process_features(dataset_train,feature_dataset_train)
        

        save_dict = {
        "X_train":X_train,
        "y_train":y_train,
        "y_preclassified_train":y_preclassified_train,
        "y_preclassified_predicted_train":y_preclassified_predicted_train,
        "X_test": X_test,
        "y_test": y_test,
        'y_preclassified_test':y_preclassified_test,
        'y_preclassified_predicted_test':y_preclassified_predicted_test
        }
        with open(save_path, 'wb') as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(save_path, 'rb') as handle:
            save_dict = pickle.load(handle)

        X_train = save_dict['X_train']
        y_train = save_dict['y_train']
        y_preclassified_train = save_dict['y_preclassified_train']
        y_preclassified_predicted_train = save_dict['y_preclassified_predicted_train']

        X_test = save_dict['X_test']
        y_test = save_dict['y_test']
        y_preclassified_test = save_dict['y_preclassified_test']
        y_preclassified_predicted_test = save_dict['y_preclassified_predicted_test']



    randforest = RandomForestClassifier(class_weight='balanced',max_features=30,n_estimators=1000)
    adaboost = AdaBoostClassifier()
    logistic = LogisticRegression()
    clfs = [randforest,adaboost,logistic]
    for clf in clfs:
        
        clf.fit(X_train, y_train)
 
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)

        y_test_comb = np.concatenate((y_test,y_preclassified_test))
        y_train_comb = np.concatenate((y_train,y_preclassified_train))
        y_pred_test_comb = np.concatenate((y_pred_test,y_preclassified_predicted_test))
        y_pred_train_comb = np.concatenate((y_pred_train,y_preclassified_predicted_train))

        accuracy_test = accuracy_score(y_test_comb,y_pred_test_comb)
        accuracy_train = accuracy_score(y_train_comb,y_pred_train_comb)
        print(f"{type(clf)}: train: {accuracy_train} test: {accuracy_test}")
    
    # precision = precision_score(y,y_pred)
    # recall = recall_score(y,y_pred)
    # confusion = confusion_matrix(y,y_pred)
    print('Done')