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

dir_0 = "D:/data/cycloData/2016/"
dir_1 = "D:/data/cycloData/2020/"
test_csv = 'save/2016-2020-test'
train_csv = 'save/2016-2020-train'
class_labels = ['nochange','removed',"added",'change',"color_change","unfit"]
laz_paths_0 = os.listdir(dir_0)
laz_paths_1 = os.listdir(dir_1)
df_dict = {}
combined = os.listdir(train_csv) + os.listdir(test_csv)
for index in range(100):
    pass



