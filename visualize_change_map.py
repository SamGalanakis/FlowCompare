import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly
import numpy as np
from utils import (
view_cloud_plotly
)
from plotly.subplots import make_subplots


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
        html.Div(id='slider-output-container'),
        dcc.Slider(
            id='multiple_slider',
            min=0.,
            max=10.,
            step=0.1,
            value=3.)
            ] ,style ={'width':'20%'}),
          html.Div([
            html.Div(id='gen-std-output-container'),
        dcc.Slider(
            id='gen_std',
            min=0.,
            max=1.,
            step=0.05,
            value=0.6),
            ] ,style ={'width':'20%'}),
         
        html.Div([
            dcc.Graph(id='graph_0', figure={}),
            dcc.Graph(id='graph_1', figure={}),
            dcc.Graph(id='graph_1_given_0', figure={}),
            dcc.Graph(id='graph_0_given_1', figure={}),
            dcc.Graph(id='gen_given_0', figure={}),
            dcc.Graph(id='gen_given_1', figure={}),
     


        ],style={ "columnCount": 3,'rowCount': 2})
        


    ])

    


    @app.callback(
    [Output('slider-output-container', 'children'),
    Output('gen-std-output-container', 'children'),
    Output(component_id='graph_0', component_property='figure'),
    Output(component_id='graph_1', component_property='figure'),
    Output(component_id='graph_0_given_1', component_property='figure'),
    Output(component_id='graph_1_given_0', component_property='figure'),
    Output(component_id='gen_given_1', component_property='figure'),
    Output(component_id='gen_given_0', component_property='figure')],
    [Input(component_id='multiple_slider', component_property='value'),
    Input(component_id='gen_std', component_property='value'),
    Input(component_id='index_selector', component_property='value')])
   

    def index_chooser(multiple,gen_std,index):
        
        index = int(index)
        
        
        print(f'Loading index {index}!')
        print(multiple)
        graph_0,graph_1,graph_0_given_1,graph_1_given_0,gen_given_1,gen_given_0 = fig_getter(index,float(multiple),float(gen_std))
       

        return f"Std multiple: {multiple}",f"Gen std: {gen_std}",graph_0,graph_1,graph_0_given_1,graph_1_given_0,gen_given_1,gen_given_0

    
       
    app.run_server(debug=True)




    