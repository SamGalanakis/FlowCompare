import os
import pandas as pd
import numpy as np
from utils import (
load_las,
compare_clouds,
extract_area,
random_subsample,
compare_clouds

)


def manual_classifier(
    point_list_df,
    file_1,
    file_2,
    class_labels,
    sample_size=2048,
    clearance=5,
    shape='cylinder',
        ):
    points1 = load_las(file_1)
    points2 = load_las(file_2)
    change_classification_list = []
    centers = []
    for index, row in point_list_df.iterrows():
        center = np.array([row['x'],row['y']])
        extraction_1 = random_subsample(extract_area(points1,center,clearance,shape),sample_size)
        extraction_2 = random_subsample(extract_area(points2,center,clearance,shape),sample_size)
        change_classification = compare_clouds(extraction_1,extraction_2,class_labels)
        change_classification_list.append(change_classification)
    point_list_df['clearance'] = clearance
    point_list_df['classification'] = change_classification_list
    return point_list_df

if __name__ == "__main__":
    dir_1 = "D:/data/cycloData/2016/"
    dir_2 = "D:/data/cycloData/2020/"
    class_labels = ['nochange','removed','change']
    point_list_dir = "D:/data/cycloData/point_lists/2016-2020/"
    classified_dir = "D:/data/cycloData/point_lists_classified/2016-2020/"
    point_list_files = [os.path.join(point_list_dir,f) for f in os.listdir(point_list_dir) if os.path.isfile(os.path.join(point_list_dir, f))]

    point_list_dfs = [pd.read_csv(x,header=None) for x in point_list_files]
    for x in point_list_dfs:
        x.columns = ['name',"x","y","z"]
    files_dir_1 = [os.path.join(dir_1,f) for f in os.listdir(dir_1) if os.path.isfile(os.path.join(dir_1, f)) and f.split(".")[-1]=='las']
    files_dir_2 = [os.path.join(dir_2,f) for f in os.listdir(dir_2) if os.path.isfile(os.path.join(dir_2, f))and f.split(".")[-1]=='las']
    files_dir_1 = sorted(files_dir_1,key=lambda x: int(os.path.basename(x).split("_")[0]))
    files_dir_2 = sorted(files_dir_2,key=lambda x: int(os.path.basename(x).split("_")[0]))
    
    for point_list_df, file_1, file_2 in zip(point_list_dfs,files_dir_1,files_dir_2):
        point_list_df_returned = manual_classifier(point_list_df,file_1,file_2,clearance=5,class_labels=class_labels)
        number_start = os.path.basename(file_1).split("_")[0]
        image_id_1 = os.path.basename(file_1).split("_")[1].split(".")[0]
        image_id_2 = os.path.basename(file_2).split("_")[1].split(".")[0]
        final_save_path = os.path.join(classified_dir,f"{number_start}_{image_id_1}_{image_id_2}.csv")
        point_list_df_returned.to_csv(final_save_path)




