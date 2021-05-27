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

    app = dash.Dash(__name__,suppress_callback_exceptions = False)

    
    index_selector_options = [{'label':key,'value':key} for key in index_range]
    global multiple
    global index
    multiple = 3.
    

    app.layout = html.Div([
        dcc.Dropdown(id= 'index_selector',
        options=index_selector_options,
        multi=False,
        value = '0',
        style ={'width':'20%'}),
         html.Div([
        dcc.Slider(
            id='multiple_slider',
            min=0.,
            max=10.,
            step=0.1,
            value=3.),
            html.Div(id='slider-output-container')] ,style ={'width':'10%'}),
        html.Div(id="hidden_div", style={'display':"none"}),
        html.Div([
            dcc.Graph(id='graph_0', figure={}),
            dcc.Graph(id='graph_1', figure={}),
            dcc.Graph(id='graph_0_given_1', figure={}),
            dcc.Graph(id='graph_1_given_0', figure={}),
            dcc.Graph(id='gen_given_1', figure={}),
            dcc.Graph(id='gen_given_0', figure={}),


        ],style={ "columnCount": 2,'rowCount': 2})
        


    ])

    

    @app.callback(
    Output('slider-output-container', 'children'),
    Input(component_id='multiple_slider', component_property='value'),
    prevent_initial_call=False)

    def set_slider(value):
        global multiple
        multiple = float(value)
        return f"std multiple: {multiple}"

    

    @app.callback(
    Output(component_id='graph_0', component_property='figure'),
    Output(component_id='graph_1', component_property='figure'),
    Output(component_id='graph_0_given_1', component_property='figure'),
    Output(component_id='graph_1_given_0', component_property='figure'),
    Output(component_id='gen_given_1', component_property='figure'),
    Output(component_id='gen_given_0', component_property='figure'),
    
    Input(component_id='index_selector', component_property='value'),
    prevent_initial_call=False)

    def index_chooser(value):
        global index
        index = int(value)
        
        
        print(f'Loading index {index}!')
        print(multiple)
        return fig_getter(index,multiple)


       
    app.run_server(debug=True)




    