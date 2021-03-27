import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly
import numpy as np
from utils import (
view_cloud_plotly
)


def visualize_change(fig_getter,index_range):

    app = dash.Dash(__name__,suppress_callback_exceptions = True)

    
    index_selector_options = [{'label':key,'value':key} for key in index_range]

    

    app.layout = html.Div([
        dcc.Dropdown(id= 'index_selector',
        options=index_selector_options,
        multi=False,
        value = '0',
        style ={'width':'20%'}),
        # dcc.Slider(
        #     id='percentile_slider',
        #     min=0,
        #     max=100,
        #     step=0.2,
        #     value=10,
        # ),
        html.Div([
            dcc.Graph(id='graph_0', figure={}),
            dcc.Graph(id='graph_1', figure={}),
            dcc.Graph(id='graph_1_given_0', figure={}),
            dcc.Graph(id='graph_0_given_1', figure={}),


        ],style={ "columnCount": 2,'rowCount': 2})
        


    ])

    

    # @app.callback(Output(component_id='graph_t2', component_property='figure'),
    # Input(component_id='percentile_slider', component_property='value'),
    # prevent_initial_call=True)

    

    @app.callback(
    Output(component_id='graph_0', component_property='figure'),
    Output(component_id='graph_1', component_property='figure'),
    Output(component_id='graph_1_given_0', component_property='figure'),
    Output(component_id='graph_0_given_1', component_property='figure'),
    Input(component_id='index_selector', component_property='value'),
    prevent_initial_call=False)

    def index_chooser(index):
        index = int(index)
        print(f'Loading index {index}!')
        return fig_getter(index)


       
    app.run_server(debug=True)




    