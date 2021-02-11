import pandas as pd
import os 
from os import listdir
from os.path import isfile, join
import numpy as np
np.random.seed(0)
clean_classified_point_csv_dir = r"D:\data\cycloData\labeled_point_lists\2016-2020"

train_out = r"D:\data\cycloData\labeled_point_lists_train\2016-2020"
test_out = r"D:\data\cycloData\labeled_point_lists_test\2016-2020"
test_out_unlabeled = r"D:\data\cycloData\point_lists_test_unlabeled\2016-2020"

names = listdir(clean_classified_point_csv_dir)
names = [x.split('.')[0] for x in names]

csv_paths = [join(clean_classified_point_csv_dir, f) for f in listdir(clean_classified_point_csv_dir) if isfile(join(clean_classified_point_csv_dir, f))]
whole_dfs = [pd.read_csv(x) for x in csv_paths]
changes = []
added = []
nochanges = []

class_labels = ['nochange','removed',"added",'change',"color_change"]
class_map_dict = {key:class_labels.index(key) for key in class_labels}
counts_per = { key:[] for key in class_labels}
class_count_dict = {key:0 for key in class_labels}
split_percent = 0.8

for df_index, (df,name) in enumerate(zip(whole_dfs,names)):
    for row_index, row in df.iterrows():
        current_class = row['classification']
        class_count_dict[current_class] +=1
train_subset_dict = {}
for key,val in class_count_dict.items():
    n_train = int(split_percent*val)
    train_ints = np.random.choice(range(0,val),n_train,replace=False).tolist()
    train_subset_dict[key] = train_ints
train_subset_dict_counts = {key:len(val) for key,val in train_subset_dict.items()}
test_subset_dict_counts = {key:class_count_dict[key] - len(val) for key,val in train_subset_dict.items()}
class_count_dict = {key:0 for key in class_labels}
for df_index, (df,name) in enumerate(zip(whole_dfs,names)):
    test_indices = []
    train_save_path = join(train_out,name+'.csv')
    test_save_path = join(test_out,name+'.csv')
    test_unlabeled_save_path = join(test_out_unlabeled,name+'.csv')
    for row_index, row in df.iterrows():
        current_class = row['classification']
        class_count_dict[current_class] +=1     
        if not (class_count_dict[current_class] in train_subset_dict[current_class]):
            test_indices.append(row_index)
    if len(test_indices)>0:
        df_test = df.iloc[test_indices].copy()
        df_test.to_csv(test_save_path,index=False)
        df_test = df_test.drop('classification',axis=1)
        df_test.to_csv(test_unlabeled_save_path,index=False)
        df_train = df.drop(test_indices)
        df_train.to_csv(train_save_path,index=False)
        assert len(df_train) + len(df_test) == len(df)
    else:
        df.to_csv(train_save_path,index=False)

        













