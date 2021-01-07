import pandas as pd
import os 
from os import listdir
from os.path import isfile, join
import numpy as np

classified_point_csv_dir = r"D:\data\cycloData\point_lists_classified\2016-2020"
clean_classified_point_csv_dir = r"D:\data\cycloData\point_lists_classified_clean\2016-2020"
clean_int_classified_point_csv_dir = r"D:\data\cycloData\point_lists_classified_clean_int\2016-2020"

names = listdir(classified_point_csv_dir)
names = [x.split('.')[0] for x in names]

onlyfiles = [join(classified_point_csv_dir, f) for f in listdir(classified_point_csv_dir) if isfile(join(classified_point_csv_dir, f))]

changes = []
added = []
nochanges = []

class_labels = ['nochange','removed',"added",'change',"color_change"]
class_map_dict = {key:class_labels.index(key) for key in class_labels}
counts_per = { key:[] for key in class_labels}

for index , csv_path in enumerate(onlyfiles):
    df = pd.read_csv(csv_path,index_col=[0])
    df = df[df['classification'] != 'unfit']
    
    save_path = join(clean_classified_point_csv_dir , os.path.basename(csv_path))
    df['name'] = [f"Point #{x}" for x in range(len(df))]
    classification_counts = df['classification'].value_counts()
    for class_type,count_list in counts_per.items():
        if class_type in classification_counts.keys():
            to_add = classification_counts[class_type]
        else:
            to_add = 0
        count_list.append(to_add)
    if len(df)>0:
        df.to_csv(save_path,index=False)
        df['classification'] = df['classification'].apply(lambda x: class_map_dict[x])
        save_path = join(clean_int_classified_point_csv_dir , os.path.basename(csv_path))
        df.to_csv(save_path,index=False)
    
   

counts_per['name'] = names

stats_df = pd.DataFrame.from_dict(counts_per)
print(stats_df.select_dtypes(np.number).sum())

stats_df.to_csv(join(clean_classified_point_csv_dir,'scene_stats.csv'), index=False)
stats_df.select_dtypes(np.number).sum().to_csv(join(clean_classified_point_csv_dir,'total_counts.csv'))




