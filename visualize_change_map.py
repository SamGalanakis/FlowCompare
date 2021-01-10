import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
from utils import (
view_cloud_plotly
)


def visualize_change(points_t1,points_t2,log_prob_t1,log_prob_t2):

    app = dash.Dash(__name__,suppress_callback_exceptions = True)

    assert isinstance(points_t1,list), "Must be lists!"
    n_squares = len(points_t1)
    global current_square_index 
    current_square_index = 0
    drop_options_grid_square = [{'label':key,'value':key} for key in range(n_squares)]

    if points_t1[0].shape[1] ==6:
        rgb1 = points_t1[current_square_index][:,3:]
        rgb2 = points_t2[current_square_index][:,:3],points_t2[current_square_index][:,3:]
    else:
        rgb1 = rgb2 = None

    app.layout = html.Div([
        dcc.Dropdown(id= 'dropdown_grid_square',
        options=drop_options_grid_square,
        multi=False,
        value = '0',
        style ={'width':'20%'}),
        dcc.Slider(
            id='percentile_slider',
            min=0,
            max=100,
            step=0.2,
            value=10,
        ),
        html.Div([
            dcc.Graph(id='graph_t1', figure=view_cloud_plotly(points_t1[current_square_index][:,:3],rgb1,show=False)),
        dcc.Graph(id='graph_t2', figure=view_cloud_plotly(points_t2[current_square_index][:,:3],rgb2,show=False))


        ],style={ "columnCount": 2})
        


    ])

    

    @app.callback(Output(component_id='graph_t2', component_property='figure'),
    Input(component_id='percentile_slider', component_property='value'),
    prevent_initial_call=True)

    def slider_callback(value):
        global current_square_index
        percentile_val = np.percentile(log_prob_t2[current_square_index],value)
        rgb = np.zeros_like(points_t2[current_square_index][:,:3])
        rgb[log_prob_t2[current_square_index]<percentile_val] = np.array([255,0,0])
        return view_cloud_plotly(points_t2[current_square_index][:,:3],rgb,show=False)

    @app.callback(
    Output(component_id='graph_t1', component_property='figure'),
    Input(component_id='dropdown_grid_square', component_property='value'),
    prevent_initial_call=True)

    def grid_chooser(value):
        global current_square_index 
        current_square_index = value
        if points_t1.shape[1] ==6:
            rgb = points_t1[current_square_index][:,3:]
        else:
            rgb = None
        figure_1=view_cloud_plotly(points_t1[current_square_index][:,:3],rgb,show=False)
        #figure_2 = view_cloud_plotly(points_t2[current_square_index][:,:3],points_t2[current_square_index][:,3:],show=False)


        return figure_1
    app.run_server(debug=True)




    