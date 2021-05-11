import os
import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))
from utils import (
load_las,
extract_area,
random_subsample,
view_cloud_plotly
)

dir_0 = "save/challenge_data/Shrec_change_detection_dataset_public/2016"
dir_1 = "save/challenge_data/Shrec_change_detection_dataset_public/2020"
test_csv = 'save/2016-2020-test'
train_csv = 'save/2016-2020-train'
class_labels = ['nochange','removed',"added",'change',"color_change","unfit"]
csv_paths_train = {int(x.split('_')[0]):os.path.join(train_csv,x) for x in os.listdir(train_csv)}
csv_paths_test = {int(x.split('_')[0]):os.path.join(test_csv,x) for x in os.listdir(test_csv)}
max_int = max(max(csv_paths_train.keys()),max(csv_paths_test.keys()))

df_dicts=[]
for index in range(max_int):
    df_list = []
    if index in csv_paths_train:
        df_train = pd.read_csv(csv_paths_train[index])
        df_list.append(df_train)
    if index in csv_paths_test:
        df_test = pd.read_csv(csv_paths_test[index])
        df_list.append(df_test)
    
    if len(df_list)>1:
        df_comb = pd.concat(df_list)
    elif len(df_list)==1:
        df_comb = df_list[0]
    else:
        continue
    df_comb['scene'] = [index]*df_comb.shape[0]
    df_dicts.append(df_comb)
df = pd.concat(df_dicts)

df.drop('name',axis=1)

laz_paths_0 = {int(x.split('_')[0]):os.path.join(dir_0,x) for x in os.listdir(dir_0)}
laz_paths_1 = {int(x.split('_')[0]):os.path.join(dir_1,x) for x in os.listdir(dir_1)}
laz_clouds_0 = {}
laz_clouds_1 = {}
df_copy = df.copy(deep=True)
new_classifications = []
for index,row in df_copy.iterrows():
    print(f'{index} of {df.shape[0]}')
    scene_num = row['scene']
    if not scene_num in laz_clouds_0:
        laz_clouds_0[scene_num] = load_las(laz_paths_0[scene_num])
        laz_clouds_1[scene_num] = load_las(laz_paths_1[scene_num])
    cloud_0 = laz_clouds_0[scene_num]
    cloud_1 = laz_clouds_1[scene_num]
    center = np.array([row['x'],row['y']])

    extract_0 = cloud_0[extract_area(cloud_0,center,1,shape='circle'),...]
    extract_1 = cloud_1[extract_area(cloud_1,center,1,shape='circle'),...]
    extract_0 = random_subsample(extract_0,10000)
    extract_1 = random_subsample(extract_1,10000)

    extract_1[:,0] += 3.5
    combined = np.concatenate((extract_0,extract_1))
    plot = view_cloud_plotly(combined[:,:3],combined[:,3:],show=False)
    plot.show()
    classification = row['classification']
    change_label =input(f'Old": {classification}, enter new:')
    if change_label == "":
        new_classifications.append(classification)
    else:
        new_classifications.append(change_label)
    
df['classification'] = new_classifications
df.to_csv('new.csv')
        






