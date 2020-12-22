import os
import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from datetime import datetime
import sys

from utils import (
load_las,
compare_clouds,
extract_area,
random_subsample,
compare_clouds,
view_cloud_plotly
)

dir_1 = "D:/data/cycloData/2016/"
dir_2 = "D:/data/cycloData/2020/"
class_labels = ['nochange','removed',"added",'change',"color_change","unfit"]
point_list_dir = "D:/data/cycloData/point_lists/2016-2020/"
classified_dir = "D:/data/cycloData/point_lists_classified/2016-2020/"
clearance = 3
point_size = 1.5
point_list_files = [os.path.join(point_list_dir,f) for f in os.listdir(point_list_dir) if os.path.isfile(os.path.join(point_list_dir, f))]
scene_numbers = [int(os.path.basename(x).split('.')[0]) for x in point_list_files]
sample_size = 100000
point_list_dfs = [pd.read_csv(x,header=None) for x in point_list_files]

point_list_dfs = {scene_num:pd.read_csv(path,header=None) for scene_num,path in zip(scene_numbers,point_list_files)}
for x in point_list_dfs.values():
    x.columns = ['name',"x","y","z"]
files_dir_1 = [os.path.join(dir_1,f) for f in os.listdir(dir_1) if os.path.isfile(os.path.join(dir_1, f)) and f.split(".")[-1]=='las']
files_dir_2 = [os.path.join(dir_2,f) for f in os.listdir(dir_2) if os.path.isfile(os.path.join(dir_2, f))and f.split(".")[-1]=='las']
files_dir_1 = sorted(files_dir_1,key=lambda x: int(os.path.basename(x).split("_")[0]))
files_dir_2 = sorted(files_dir_2,key=lambda x: int(os.path.basename(x).split("_")[0]))

classified_point_list_files = [os.path.join(classified_dir,f) for f in os.listdir(classified_dir) if os.path.isfile(os.path.join(classified_dir, f))]

classified_point_list_files = {int(os.path.basename(x).split("_")[0]):x for x in classified_point_list_files}


def create_plots(point_list_df,file_1,file_2,sample_size=2048,clearance=2,shape='cylinder'):
    points1 = load_las(file_1)
    points2 = load_las(file_2)
    current_figure_tuples = []
    for index, row in point_list_df.iterrows():
        center = np.array([row['x'],row['y']])
        #Remove ground
        # points1 = points1[points1[:,2]>0.43]
        # points2 = points2[points2[:,2]>0.43]
        extraction_1 = random_subsample(extract_area(points1,center,clearance,shape),sample_size)
        extraction_2 = random_subsample(extract_area(points2,center,clearance,shape),sample_size)
        fig_1 = view_cloud_plotly(extraction_1[:,:3],extraction_1[:,3:],show=False,point_size=point_size)
        fig_2 = view_cloud_plotly(extraction_2[:,:3],extraction_2[:,3:],show=False,point_size=point_size)
        current_figure_tuples.append((fig_1,fig_2))
    return current_figure_tuples





external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__,external_stylesheets=external_stylesheets,suppress_callback_exceptions = True) 


drop_options_scene = [{'label':key,'value':key} for key in scene_numbers]


current_classifications = {}

for scene_number,classified_point_list_file in classified_point_list_files.items():
    classified_point_list_df = pd.read_csv(classified_point_list_file)

    current_classifications[scene_number] = classified_point_list_df['classification'].tolist()

app.layout = html.Div([
dcc.Dropdown(id= 'scene_number',
options=drop_options_scene,
multi=False,
value = '0',
style ={'width':'20%'}),
html.Div(id='once_chosen_scene_div'),

html.Div(id='placeholder', style={"display":"none"}),
html.Div(id="placeholder2", style={"display":"none"}),


])

@app.callback(
    Output(component_id='once_chosen_scene_div', component_property='children'),
    Input(component_id='scene_number', component_property='value'),prevent_initial_call=True
)


def scene_changer(scene_number):
    global sample_size
    global clearance
    global current_figure_tuples
    global n_points
    global current_classifications 
    global current_point_number
    global current_scene_number
    global drop_options_classifiation

    #Start at first point for all scenes
    current_point_number=0
    current_scene_number = i =  int(scene_number)
    
    n_points = point_list_dfs[i].shape[0]

    if not scene_number in current_classifications.keys():
        current_classifications[i] = ['UNSET']*n_points

    
    #Update plot list
    current_figure_tuples = create_plots(point_list_dfs[i],files_dir_1[i],files_dir_2[i],sample_size=sample_size,clearance=clearance)


    
    drop_options_classifiation = [{'label':key,'value':key} for key in class_labels]
    drop_options_point_number = [{'label':key,'value':key} for key in range(point_list_dfs[i].shape[0])]
    
    once_chosen_scene_div = html.Div([

    html.Button("Save scene", id="savebtn",n_clicks=0),
    dcc.Dropdown(id= 'point_number',
    options=drop_options_point_number,
    multi=False,
    value = '0',
    style ={'width':'20%'}),

    html.Div(id='specific_point',children=[html.Div([
    dcc.Dropdown(id= 'classification',
    options=drop_options_classifiation,
    multi=False,
    value = current_classifications[i][current_point_number],
    style ={'width':'30%'}),


        
            html.H3('Time 1'),
            dcc.Graph(id='g1', figure=current_figure_tuples[0][0])
        ], className="six columns"),

        html.Div([
            html.H3('Time 2'),
            dcc.Graph(id='g2', figure=current_figure_tuples[0][1])
        ], className="six columns"),
    ], className="row")
    ])


    return once_chosen_scene_div


@app.callback(
    # Output(component_id='specific_point', component_property='children'),
    Output(component_id='g1', component_property='figure'),
    Output(component_id='g2', component_property='figure'),
    Output(component_id='classification', component_property='value'),
    Input(component_id='point_number', component_property='value'),prevent_initial_call=True
)


def point_changer(point_number):


    global current_point_number
    global current_scene_number
    global drop_options_classifiation
    current_point_number = int(point_number)
     
    fig_1 = current_figure_tuples[current_point_number][0]
    fig_2 = current_figure_tuples[current_point_number][1]
    classification_of_new_points = current_classifications[current_scene_number][current_point_number]
   
    
    return fig_1,fig_2,classification_of_new_points



@app.callback(
    Output(component_id='placeholder', component_property='children'),
    Input(component_id='classification', component_property='value'),prevent_initial_call=True)


def classification_changer(input):
    global current_classifications
    global current_point_number
    global current_scene_number
    print(f"Changing classification of scene {current_scene_number} point {current_point_number}")
    current_classifications[current_scene_number][current_point_number] = input



@app.callback(
    Output(component_id='placeholder2', component_property='children'),
    Input(component_id='savebtn', component_property='n_clicks'),prevent_initial_call=True
)

def saver(input):
    global current_classifications
    global current_point_number
    global current_scene_number

    point_list_df = point_list_dfs[current_scene_number].copy()
    point_list_df['classification'] = current_classifications[current_scene_number]
    number_start = current_scene_number
    file_1 = files_dir_1[current_scene_number]
    file_2 = files_dir_1[current_scene_number]
    image_id_1 = os.path.basename(file_1).split("_")[1].split(".")[0]
    image_id_2 = os.path.basename(file_2).split("_")[1].split(".")[0]
    final_save_path = os.path.join(classified_dir,f"{number_start}_{image_id_1}_{image_id_2}.csv")
    point_list_df.to_csv(final_save_path)
    print("Saving file!")
    
    

app.run_server(debug=True)










