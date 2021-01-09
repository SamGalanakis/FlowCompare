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



    


    app.layout = html.Div([
        dcc.Slider(
            id='percentile_slider',
            min=0,
            max=100,
            step=0.2,
            value=10,
        ),
        html.Div([
            dcc.Graph(id='graph_t1', figure=view_cloud_plotly(points_t1,show=False)),
        dcc.Graph(id='graph_t2', figure=view_cloud_plotly(points_t2,show=False))


        ],style={ "columnCount": 2})
        


    ])

    
    @app.callback(Output(component_id='graph_t2', component_property='figure'),
    Input(component_id='percentile_slider', component_property='value'),
    prevent_initial_call=True)

    def slider_callback(value):
        percentile_val = np.percentile(log_prob_t2,value)
        to_plot = points_t2[log_prob_t2>percentile_val]
        rgb = np.zeros_like(points_t2)
        rgb[log_prob_t2<percentile_val] = np.array([255,0,0])
        return view_cloud_plotly(points_t2,rgb,show=False)


    app.run_server(debug=True)




    