# -*- coding: utf-8 -*-
"""
Created on Fri May  7 12:45:26 2021

@author: danilo
"""

#scatter plots with ranger slider
import pandas as pd     #(version 0.24.2)

import dash             #(version 1.0.0)
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import plotly.express as px
import flask
import glob
import os


df = pd.read_csv("IST_Central_Pav_Clean0.csv")
#df= df.drop(columns=['Unnamed: 0'])
# Dashboard implementation
#raw_data_tot= pd.read_csv('IST_Central_Pav_Clean0.csv')
# print(raw_data_tot.columns)
# df=raw_data_tot

# Organizaion of Dashboard in tabs
external_stylesheets = ['https://raw.githubusercontent.com/STATWORX/blog/master/DashApp/assets/style.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

image_directory1 ='/Users/danilo/anaconda3/envs/dashboards/Dashboards/assets/clustering_image/'
images_list1 = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory1))]
# image_directory2 ='/Users/danilo/anaconda3/envs/dashboards/Dashboards/assets/featureSelection_image/'
# images_list2 = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory2))]
# image_directory3 ='/Users/danilo/anaconda3/envs/dashboards/Dashboards/assets/Regression_image/'
# images_list3 = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory3))]
# static_image_route1 = '/static/'
static_image_route = '/static/'




app.layout = html.Div([
    html.H2('IST Energy Yearly Consumption (kWh)'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw data', value='tab-1'),
        dcc.Tab(label='Exploratory data analysis', value='tab-2'),
        dcc.Tab(label='Clustering', value='tab-3'),
        dcc.Tab(label='Feature selection', value='tab-4'),
        dcc.Tab(label='Regression methods', value='tab-5'),
    ]),
    html.Div(id='tabs-content')
])



## EDA -> Data visuliazion
## Interactive Table
tab1_layout = html.Div([
               html.H2('Data cleaned table'),
               dash_table.DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": i, "id": i, "deletable": True, "selectable": True, "hideable": True}
            
            for i in df.columns
        ],
        data=df.to_dict('records'),  # the contents of the table        
        row_selectable="multi",     # allow users to select 'multi' or 'single' rows
        row_deletable=True,         # choose if user can delete a row (True) or not (False)           
        selected_rows=[],           # indices of rows that user selects
        page_action="native",       # all data is passed to the table up-front or not ('none')
        page_current=0,             # page number that user is on
        page_size=24,                # number of rows visible per page
        style_cell={                # ensure adequate header width when text is shorter than cell's text
            'minWidth': 95, 'maxWidth': 95, 'width': 95
        },
       
        style_data={                # overflow cells' content into multiple lines
            'whiteSpace': 'normal',
            'height': 'auto'
        }
    ),

    html.Br(),
    html.Br(),
    html.Div(id='bar-container')

])

@app.callback(
    Output(component_id='bar-container', component_property='children'),
    [Input(component_id='datatable-interactivity', component_property="derived_virtual_data"),
      Input(component_id='datatable-interactivity', component_property='derived_virtual_selected_rows'),
      Input(component_id='datatable-interactivity', component_property='selected_rows')]
    
)
def update_bar(all_rows_data, slctd_row_indices, slctd_rows):
              
    dff = pd.DataFrame(all_rows_data)
    dff=dff.groupby(['Hour'],as_index=False)['Power[kW]'].mean()

    # used to highlight selected countries on bar chart
    colors = ['#7FDBFF' if i in slctd_row_indices else '#0074D9'
              for i in range(len(dff))]

    if "Power[kW]" in dff :
        return [
            dcc.Graph(id='bar-chart',
                      figure=px.bar(
                          data_frame=dff,
                          x='Hour',
                          y='Power[kW]',
                         
                      )
                      .update_traces(marker_color=colors, hovertemplate="<b>%{y}%</b><extra></extra>")
                      )
        ]

#Exploratory data analysis
    
tab2_layout =  html.Div( children=[
               html.H3('Time series graph'),
             dcc.Graph(
                id="yearly-data",
                figure={
                    "data":[
                        {'x': df['Date'], 'y': df['Power[kW]'], 'type': 'line', 'name': 'Power'},
                        {'x': df['Date'], 'y': df['Temperature[C]'], 'type': 'line', 'name': 'Temperature'},

                        ],
                   'layout': {
                       'title': 'Power consumption vs Temperature'
            }
        }
    ),
            
              dcc.Graph(
                id="yearly-data",
                figure={
                    "data":[
                        {'x':df['Date'], 'y': df['Solar_Radiation[W/m^2]'], 'type': 'line', 'name': 'Solar radiation'},
                        {'x': df['Date'], 'y': df['Power[kW]'], 'type': 'line', 'name': 'Power'},

                        ],
                   'layout': {
                       'title': 'Power consumption vs Solar Radiation'
            }
        }
    ),
    
           
    
#---------------------------------------------------------------    
    
#Violin plots


    html.H1('Violin Plots'),
    html.Hr(),
    html.Div(className='two columns', children=[
        dcc.RadioItems(
            id='items',
            options=[
                {'label': 'Power', 'value': 'Power[kW]'},
                {'label': 'temperature', 'value': 'Temperature[C]'},
                {'label': 'Solar radiation', 'value': 'Solar_Radiation[W/m^2]'}
                
            ],
            value='Power[kW]',
            style={'display': 'block'}
        ),
        html.Hr(),
        dcc.RadioItems(
            id='points',
            options=[
                {'label': 'Display All Points', 'value': 'all'},
                {'label': 'Hide Points', 'value': False},
                {'label': 'Display Outliers', 'value': 'outliers'},
                {'label': 'Display Suspected Outliers', 'value': 'suspectedoutliers'},
            ],
            value='all',
            style={'display': 'block'}
        ),
        
    ]),
    html.Div(dcc.Graph(id='graph'), className='ten columns')
])

@app.callback(
    Output('graph', 'figure'), [
    Input('items', 'value'),
    Input('points', 'value')])
def update_graph(value, points):
    return {
        'data': [
            {
                'type': 'violin',
                'y': df[value],
                'text': ['Sample {}'.format(i) for i in range(len(df))],
                'points': points,
              
            }
        ],
        'layout': {
            'margin': {'l': 30, 'r': 10, 'b': 30, 't': 0}
        }
    }

#---------------------------------------------------------------

#Clustering

df = df.set_index(pd.to_datetime(df['Date']))                   # make Date into 'datetime' and then index
df = df.drop (columns = 'Date')
df['year'] = df.index.year 
df['days']  = df.index.day
df=df[df['year']==2017]




mark_values = {1:'January',2:'February',3:'March',4:'April',
               5:'May',6:'June',7:'July',8:'August',
               9:'September',10:'October',11:'November',12:'December'}


tab3_layout = html.Div([
              html.Div([
              html.Div([
              html.H2(children= "Scatter plots",
              style={"text-align": "center", "font-size":"100%", "color":"blue"}),
              html.H4("The scatter plots below are represented by the most important "
                    "model feature (Power-Temperature-Solar Radiation)")
            
        ]),

        html.Div([
            dcc.Graph(id='the_graph1')
        ]),
        
         html.Div([
            dcc.Graph(id='the_graph2')
        ]),


        html.Div([
            dcc.RangeSlider(id='slider',
                min=1,
                max=12,
                value=[1],
                marks=mark_values)
        ],style={"width": "80%", "position":"absolute",
                 "left":"10%"}),



]),         
 
     html.Div([html.H4('Clustering results'),
     dcc.Dropdown(
        id='image_clust',        
        options=[{'label':i, 'value':i} for i in images_list1],
        value=images_list1[0]
    ),
     html.Img(id='outimage')
])              
   
])                                
                 
                 
                 
#---------------------------------------------------------------
                 
                 

@app.callback(
    Output('the_graph1','figure'),
    [Input('slider','value')]
)

def update_graph1(value):
    dff=df[(df['Months']>=value[0])]
   
    # filter df rows where column year values are >=1985 AND <=1988
    dff=dff.groupby(["days"], as_index=False)["Temperature[C]",
                    "Power[kW]"].mean()
    # print (dff[:3])

    scatterplot1 = px.scatter(
        data_frame=dff,
        x="Power[kW]",
        y="Temperature[C]",
        text="days",
        height=550
    )

    scatterplot1.update_traces(textposition='top center')

    return (scatterplot1)

@app.callback(
    Output('the_graph2','figure'),
    [Input('slider','value')]
)

def update_graph2(value):
    dff=df[(df['Months']>=value[0])]
   
    # filter df rows where column year values are >=1985 AND <=1988
    dff=dff.groupby(["days"], as_index=False)["Solar_Radiation[W/m^2]",
                    "Power[kW]"].mean()
    # print (dff[:3])

    scatterplot2 = px.scatter(
        data_frame=dff,
        x="Power[kW]",
        y="Solar_Radiation[W/m^2]",
        text="days",
        height=550
    )

    scatterplot2.update_traces(textposition='top center')

    return (scatterplot2)




@app.callback(
    dash.dependencies.Output('outimage','src'),
    [dash.dependencies.Input('image_clust','value')])
def update_image_src(value):
    return static_image_route + value
@app.server.route('{}<image_path>.png'.format(static_image_route))
def serve_image(image_path):
    image_name = '{}.png'.format(image_path)
    if image_name not in images_list1:
        raise Exception ('"{}" is not included'.format(image_path))
    return flask.send_from_directory(image_directory1,image_name)



# Feature selection

# tab4_layout =html.Div([html.H2('Feature selection results'),
#     html.Label('Checkboxes'),
#     dcc.Checklist(
#         id='image_feat',
#         options=[{'label':i, 'value':i} for i in images_list1],
#         value=images_list1[0]
#     ),
#      html.Img(id='outimage2')
# ])              



# @app.callback(
#     dash.dependencies.Output('outimage2','src'),
#     [dash.dependencies.Input('image_feat','value')])
# def update_image_src1(value):
#     return static_image_route2 + value
# @app.server.route('{}<image_path>.png'.format(static_image_route2))
# def serve_image1(image_path):
#     image_name2 = '{}.png'.format(image_path)
#     if image_name2 not in images_list2:
#         raise Exception ('"{}" is not included'.format(image_path))
#     return flask.send_from_directory(image_directory2,image_name2)


## Regression model

# tab5_layout =html.Div([html.H2('Regression model results'),
#      dcc.Dropdown(
#         id='image_clust',        
#         options=[{'label':i, 'value':i} for i in images_list3],
#         value=images_list3[0]
#     ),
#      html.Img(id='outimage')
# ])

# @app.callback(
#     dash.dependencies.Output('outimage','src'),
#     [dash.dependencies.Input('image_clust','value')])
# def update_image_src2(value):
#     return static_image_route + value
# @app.server.route('{}<image_path>.png'.format(static_image_route))
# def serve_image2 (image_path):
#     image_name = '{}.png'.format(image_path)
#     if image_name not in images_list3:
#         raise Exception ('"{}" is not included'.format(image_path))
#     return flask.send_from_directory(image_directory3,image_name)











# -------------------------------------------------------------------------------------
@app.callback (Output('tabs-content', 'children'),
              Input('tabs', 'value'))



def render_content(tab):
     if tab == 'tab-1':
        return tab1_layout
     elif tab == 'tab-2':
        return tab2_layout
     elif tab == 'tab-3':
        return tab3_layout
     # elif tab == 'tab-4':
     #    return tab4_layout
     # elif tab == 'tab-5':
     #    return tab5_layout
   






if __name__ == '__main__':
    app.run_server(debug=False)
