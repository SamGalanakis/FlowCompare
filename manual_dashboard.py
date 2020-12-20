import os
import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

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
class_labels = ['nochange','removed',"added",'change',"color_change"]
point_list_dir = "D:/data/cycloData/point_lists/2016-2020/"
classified_dir = "D:/data/cycloData/point_lists_classified/2016-2020/"
clearance = 3
point_list_files = [os.path.join(point_list_dir,f) for f in os.listdir(point_list_dir) if os.path.isfile(os.path.join(point_list_dir, f))]
sample_size = 50000
point_list_dfs = [pd.read_csv(x,header=None) for x in point_list_files]
for x in point_list_dfs:
    x.columns = ['name',"x","y","z"]
files_dir_1 = [os.path.join(dir_1,f) for f in os.listdir(dir_1) if os.path.isfile(os.path.join(dir_1, f)) and f.split(".")[-1]=='las']
files_dir_2 = [os.path.join(dir_2,f) for f in os.listdir(dir_2) if os.path.isfile(os.path.join(dir_2, f))and f.split(".")[-1]=='las']
files_dir_1 = sorted(files_dir_1,key=lambda x: int(os.path.basename(x).split("_")[0]))
files_dir_2 = sorted(files_dir_2,key=lambda x: int(os.path.basename(x).split("_")[0]))






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
        fig_1 = view_cloud_plotly(extraction_1[:,:3],extraction_1[:,3:],show=False)
        fig_2 = view_cloud_plotly(extraction_2[:,:3],extraction_2[:,3:],show=False)
        current_figure_tuples.append((fig_1,fig_2))
    return current_figure_tuples





external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__,external_stylesheets=external_stylesheets,suppress_callback_exceptions = True) 


drop_options_scene = [{'label':key,'value':key} for key in range(len(point_list_dfs))]







app.layout = html.Div([
dcc.Dropdown(id= 'scene_number',
options=drop_options_scene,
multi=False,
value = '0',
style ={'width':'40%'}),
html.Div(id='once_chosen_scene_div'),

])

@app.callback(
    Output(component_id='once_chosen_scene_div', component_property='children'),
    Input(component_id='scene_number', component_property='value')
)


def scene_changer(scene_number):
    global sample_size
    global clearance
    global current_figure_tuples
    global n_points
    
    i = int(scene_number)
    n_points = point_list_dfs[i].shape[0]
    #Update plot list
    current_figure_tuples = create_plots(point_list_dfs[i],files_dir_1[i],files_dir_2[i],sample_size=sample_size,clearance=clearance)


    n_points_now = point_list_dfs[i].shape[0]
    drop_options_classifiation = [{'label':key,'value':key} for key in class_labels]
    drop_options_point_number = [{'label':key,'value':key} for key in range(point_list_dfs[i].shape[0])]
    
    once_chosen_scene_div = html.Div([

    dcc.Dropdown(id= 'point_number',
    options=drop_options_point_number,
    multi=False,
    value = '0',
    style ={'width':'40%'}),


    dcc.Dropdown(id= 'classification',
    options=drop_options_classifiation,
    multi=False,
    value = 'nochange',
    style ={'width':'40%'}),

    # dcc.Graph(id='graph1',figure=current_figure_tuples[0][0]),
    # dcc.Graph(id='graph2',figure=current_figure_tuples[0][1]),


    html.Div(id='graph_row',
        children=[html.Div([
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
    Output(component_id='graph_row', component_property='children'),
    Input(component_id='point_number', component_property='value')
)


def point_changer(point_number):



    point_number = int(point_number)


    new_graph_row =  html.Div([


    html.Div([
    html.H3('Time 1'),
    dcc.Graph(id='g1', figure=current_figure_tuples[point_number][0])
    ], className="six columns"),

    html.Div([
    html.H3('Time 2'),
    dcc.Graph(id='g2', figure=current_figure_tuples[point_number][1])
    ], className="six columns")


    ])
    
    return new_graph_row

app.run_server(debug=True)










