# -*- coding: utf-8 -*-
"""
Created on Mon May 17 18:28:48 2021

@author: danilo
"""


# Import libraries

import pandas as pd                           # Allows reading, writing and handling data.
        

# Import libraries for Dashboard
import dash  
from dash.dependencies import Input, Output
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
import base64

#Import libraries for feature selection, regression models and clustering     # score metrix

from sklearn.cluster import KMeans

# Import Data already cleaned
raw_data_tot= pd.read_csv('IST_Central_Pav_Clean0.csv') #Import data

df= raw_data_tot
df = df.set_index(pd.to_datetime(df['Date'])) 
df['Hour'] = df.index.hour 
df.rename(columns = {'Solar_Radiation[W/m^2]': 'Solar_Rad[W/m^2]', 'Relative_Humidity': 'RH'}, inplace = True)

# Dashboard implementation

# Application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Server
server= app.server

# Adding image for clustering   
image_dpattern= 'assets/clust.png'
encoded_image_dailyp = base64.b64encode(open(image_dpattern, 'rb').read())

# Sidebar style definition
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# Padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div([
          html.Center(
        (html.Img(src= 'https://www.eduopinions.com/wp-content/uploads/2018/05/InstitutoSuperiorTecnico-logo.png',
        style = {'height':'11%', 'width':'30%' , 'text-align':'center'}))),
        
        html.H4('Central Building Consumption forecast ', className="display-5",
                style={'color':'blue', 'text-align':'center'}),
        html.Hr(),
        html.P ('Angelo Danilo Nuzzolo ist1100830', style={'text-align':'center'}),
        html.Hr(),
        
        dbc.Nav(
            [
                dbc.NavLink("Raw data display", href="/", active="exact"),
                dbc.NavLink("EDA 1", href="/page-1", active="exact"),
                dbc.NavLink("EDA 2", href="/page-2", active="exact"),
                 dbc.NavLink("EDA 3", href="/page-3", active="exact"),
                dbc.NavLink("Feature selection", href="/page-4", active="exact"),
                dbc.NavLink("Regression model", href="/page-5", active="exact"),
                dbc.NavLink("Clustering", href="/page-6", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)


content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

# Organization of dashboard in tabs
app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content

])

##  RAW DATA VISUALIZATION : Interactive Table 
pag1_layout = html.Div([
               html.H3('Data cleaned table',
               style={'color':'white','text-align':'center'}),
               html.P('Interactive table representative data cleaned'),
               dash_table.DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": i, "id": i, "selectable": True, "hideable": True}
            
            for i in df.columns
        ],
        data=df.to_dict('records'),  # the contents of the table        
        filter_action="none",       # allow filtering of data by user ('native') or not ('none')
        sort_action="native",       # enables data to be sorted per-column by user or not ('none')
        sort_mode="single",         # sort across 'multi' or 'single' columns
        row_selectable="multi",     # allow users to select 'multi' or 'single' rows
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

#----------------------------------------------------------------------


# EXPLORATORY DATA ANALYSIS (Raw data cleaned)
    # The EDA is a philosophy to explore data (mostly with visual representation) and propose new hypothesis.


## Comments : 
    # It can be noticed a positive correlation between temperature and power, this means that an increasing of temperature leads to an increase in power consumption (air conditioners or fans are turned on).
    # In the case of Solar Radiation it is possible to see a important positive correlation, this means that the trade off between the possibility to turning the light off (saving energy) and the increasing of temperature is won by the latter.
    # The negative correlation between RH and power is explained by the necessity to humidify the air with an air conditioner when the RH drops below 40%.
    # The correlation between Holiday and Power is negative beacuse people are generally not at home during the Holidays.




## EDA 1  

# Time series graph

pag2_layout =  html.Div([
               html.H3('Time series graph',
                       style={'color':'white', 'text-align':'center'}),
               html.P('Select between three different time series graph:'),
              dcc.Dropdown(
                  id ='dropd-tm',
                  options=[
                      {'label': 'Power consumption vs Temperature', 'value':111},
                      {'label': 'Power consumption vs Solar Radiation', 'value':112},
                      {'label': 'Power consumption vs Relative humidity', 'value':113},
                      ],
                  value=111
                  ),
              html.Div (id='tm')
])


@app.callback(Output('tm', 'children'),
              Input('dropd-tm', 'value'))
def render_reg (ts):
                      
                      
    if ts==111:
        return html.Div([
                dcc.Graph(
                figure={
                    "data":[
                        {'x': df.index, 'y': df['Power[kW]'], 'type': 'line', 'name': 'Power'},
                        {'x': df.index, 'y': df['Temperature[C]'], 'type': 'line', 'name': 'Temperature'},

                        ],
                   'layout': {
                       'title': 'Power consumption vs Temperature'
            }
        }
    ),
           
             
             ])
              
                       
    if ts==112:
        return html.Div([   
               dcc.Graph(
                figure={
                    "data":[
                        {'x':df.index, 'y': df['Solar_Rad[W/m^2]'], 'type': 'line', 'name': 'Solar radiation'},
                        {'x': df.index, 'y': df['Power[kW]'], 'type': 'line', 'name': 'Power'},

                        ],
                   'layout': {
                       'title': 'Power consumption vs Solar Radiation'
            }
        }
    ),
    
    ])
              
                       
    if ts==113:
        return html.Div([       
              dcc.Graph(
                figure={
                    "data":[
                        {'x':df.index, 'y': df['RH'], 'type': 'line', 'name': 'Relative Humidity'},
                        {'x': df.index, 'y': df['Power[kW]'], 'type': 'line', 'name': 'Power'},

                        ],
                   'layout': {
                       'title': 'Power consumption vs Relative Humidity'
            }
        }
    )
    
              ])
              
            
 
#Comments:
        # It is clearly possible to notice power peaks at low and high temperaure, this allow us to understand that the building is heated and cooled electrically, with greater consumption during the summer months.
        # During the holidays the consumption is low.
        # The 'trade off' between the high consumption during the night due to the switchig lihts on , and the high consumption at high solar radiation during the summer months is clearly visible.

## EDA 2   
# Violin plots, to visualize outliers

#Notes:
      # Violin plots are similar to box plots, except that they also show the probability density of the data at different values.
    

pag3_layout=html.Div([
    html.H3('Violin Plots',
    style={'color':'white', 'text-align':'center'}),
    html.H6('Outliers detection'),
    html.Hr(),
    html.Div(className='two columns', children=[
        dcc.RadioItems(
            id='items',
            options=[
                {'label': 'Power', 'value': 'Power[kW]'},
                {'label': 'Temperature', 'value': 'Temperature[C]'},
                {'label': 'Solar radiation', 'value': 'Solar_Rad[W/m^2]'}
                
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


                 

#Callbacks for violin plots               

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



## Discover outliers with mathematical function.
## In this section is removed the outliers of power consumption.

# Clean data from outliers with IQR

Q1 = df['Power[kW]'].quantile(0.25)                                                                            # First quartile
Q3 =df['Power[kW]'].quantile(0.75)                                                                              # Third quartile
IQR = Q3- Q1                                                                                                               # Inter quartile range

rdt_clean =df[((df['Power[kW]'] > (Q1 - 1.5*IQR)) & (df['Power[kW]'] < (Q3 + 1.5 *IQR)))]   # we accept all the data in the interval
# Comments :
    # The points outside the interval are deleted.
 
    
 
#------------------------------------------------------------------------------ 

## EDA 3

## In this section is evaluated the corellation between power, temperature and solar radiation each day of the year.

## Dashboard 

dfe=rdt_clean
dfe = dfe.set_index(pd.to_datetime(dfe['Date']))                   # make Date into 'datetime' and then index
dfe = dfe.drop (columns = 'Date')
dfe['year'] = dfe.index.year
dfe['Months'] = dfe.index.month  
dfe['days']  = dfe.index.day
dfe=dfe[dfe['year']==2017]




mark_values = {1:'January',2:'February',3:'March',4:'April',
               5:'May',6:'June',7:'July',8:'August',
               9:'September',10:'October',11:'November',12:'December'}


pag4_layout = html.Div([
              html.Div([
              html.H3("Scatter plots",
              style={"text-align": "center", "color":"white"}),
              html.H6("The scatter plots show the daily correlation between power,tempearture and solar radiation ")
            
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
        ],style={"width": "65%", "position":"absolute",
                 "left":"30%"})

])

#Callbacks for scatter plots                 
@app.callback(                        
    Output('the_graph1','figure'),
    [Input('slider','value')]
)

def update_graph1(value):
    dff=dfe[(dfe['Months']>=value[0])]
   
   
    dff=dff.groupby(["days"], as_index=False)["Temperature[C]",
                    "Power[kW]"].mean()
   

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
    dff=dfe[(dfe['Months']>=value[0])]
   
    # filter df rows where column year values are >=1985 AND <=1988
    dff=dff.groupby(["days"], as_index=False)["Solar_Rad[W/m^2]",
                    "Power[kW]"].mean()
    # print (dff[:3])

    scatterplot2 = px.scatter(
        data_frame=dff,
        x="Power[kW]",
        y="Solar_Rad[W/m^2]",
        text="days",
        height=550
    )

    scatterplot2.update_traces(textposition='top center')

    return (scatterplot2)
       



#----------------------------------------------------------------------

# FEATURE SELECTION AND ENGINEERING:
    # It consist in choosing the set of relevant variables that can be used to develop a model.
    # It is used to improve the accuracy, avoid overfitting and reduce the complexity of the model.
    
ddf= rdt_clean
ddf = ddf.drop (columns = 'Date')
ddf['Months'] = ddf.index.month 
ddf['Week Day'] = ddf.index.dayofweek                                  # Add day of week column



##To calculate the power at a certain time instant, we need to know the power at time t-1
ddf['Power-1']=ddf['Power[kW]'].shift(1)                               # Previous hour consumption
ddf=ddf.dropna()                                                       # Drop NaN values


## Dashboard


pag5_layout =html.Div([
             html.H3('Feature selection',
             style={'color':'white', 'text-align':'center'}),
             html.H6('The best features are the ones with higher score'),
             html.P('Select the "Feature selection method":'),
            dcc.RadioItems(
            id='radio',
            options=[
            {'label': 'f-test ANOVA', 'value': 1},
            {'label': 'Mutual information', 'value': 2},
            {'label': 'Random Forest Regressor', 'value': 3},
            ],
            value=1
                  ),
              html.Div (id='feat_select'),
              ])


@app.callback(Output('feat_select', 'children'),
              Input('radio', 'value'))
def render_feat (feature):
                      
                      
    if feature==1:
        return html.Div([
              dcc.Graph(
        figure={
            'data': [
            {'x': ['Temperature[C]'], 'y': [489.30025279], 'type': 'bar', 'name': 'Temperature[C]'},
            {'x': ['RH'], 'y': [640.97865389], 'type': 'bar', 'name': 'Relative_Humidity'},
            {'x': ['Solar_Rad[W/m^2]'], 'y': [3344.74010083], 'type': 'bar', 'name': 'Solar_Rad[W/m^2]'},
            {'x': ['Holiday'], 'y': [157.37249663], 'type': 'bar', 'name': 'Holiday'},
            {'x': ['Hour'], 'y': [257.94354102], 'type': 'bar', 'name': 'Hour'},
            {'x': ['Months'], 'y': [85.87449002], 'type': 'bar', 'name': 'Months'},
            {'x': ['Week Day'], 'y': [1017.49989346], 'type': 'bar', 'name': 'Week Day'},
            {'x': ['Power-1'], 'y': [77226.12283949], 'type': 'bar', 'name': 'Power-1'},
                        
               
           ],
            'layout': {
                'title': 'f-test ANOVA'
            }
        }
    ),])
              
                       
    if feature==2:
        return html.Div([   
                dcc.Graph(
        figure={
            'data': [
            {'x': ['Temperature[C]'], 'y': [0.11264299], 'type': 'bar', 'name': 'Temperature[C]'},
            {'x': ['RH'], 'y': [0.07286712], 'type': 'bar', 'name': 'Relative_Humidity'},
            {'x': ['Solar_Rad[W/m^2]'], 'y': [0.28516043], 'type': 'bar', 'name': 'Solar_Rad[W/m^2]'},
            {'x': ['Holiday'], 'y': [0.0216017], 'type': 'bar', 'name': 'Holiday'},
            {'x': ['Hour'], 'y': [0.52246417], 'type': 'bar', 'name': 'Hour'},
            {'x': ['Months'], 'y': [0.10762179], 'type': 'bar', 'name': 'Months'},
            {'x': ['Week Day'], 'y': [0.17243253], 'type': 'bar', 'name': 'Week Day'},
            {'x': ['Power-1'], 'y': [1.43884488], 'type': 'bar', 'name': 'Power-1'},
                        
            ],
            'layout': {
                'title': 'Mutual information'
            }
        }
    ),])
              
                       
    if feature==3:
        return html.Div([       
               dcc.Graph(
        figure={
            'data': [
            {'x': ['Temperature[C]'], 'y': [3.84010066e-03], 'type': 'bar', 'name': 'Temperature[C]'},
            {'x': ['RH'], 'y': [2.58759955e-03], 'type': 'bar', 'name': 'Relative_Humidity'},
            {'x': ['Solar_Rad[W/m^2]'], 'y': [3.99018106e-03], 'type': 'bar', 'name': 'Solar_Rad[W/m^2]'},
            {'x': ['Holiday'], 'y': [2.26997734e-04], 'type': 'bar', 'name': 'Holiday'},
            {'x': ['Hour'], 'y': [9.52311771e-02], 'type': 'bar', 'name': 'Hour'},
            {'x': ['Months'], 'y': [2.38661804e-03], 'type': 'bar', 'name': 'Months'},
            {'x': ['Week Day'], 'y': [3.55102266e-03], 'type': 'bar', 'name': 'Week Day'},
            {'x': ['Power-1'], 'y': [8.88186303e-01], 'type': 'bar', 'name': 'Power-1'},
                        
            ],
            'layout': {
                'title': 'Random Forest Regressor'
            }
        }
    )
    
              
              
])

## Final Result:
    # Power-1, Hour, Solar radiation,Week day, Temperature are the most important features, they will be included in the model.





#----------------------------

# REGRESSION

## Pre-processing 


#----------------------------------------------------------------------


# Extreme Gradient Boosting :
    # XGBoost is basically designed to enhance the performance and speed of a Machine Learning model, it is a Gradient boosting with second order derivative.



# Results:
    # This model appears to be the better one.
    # The MAE= 7.02 [kW] (here the error is blind by the compensation of negative and positive ones), MSE= 122.0 [kW]^2, RMSE= 11.05 [kW] ( for this metric don't care if the error is negative or positive, it is always a error), cvRMSE= 5.3 [%]
    # The randomness of the process might be change the values above.

#--------------------------------------------------------------------------------

# Bootstrapping :
    # It is a statistical resampling techniques that involves random sampling of a dataset with replacement (it is possible to have same sample belonging to the same dataset because there is not a division but a replacement)
    # It consist on creating multiple datasets from the original one, develop indipendet learners for each dataset and aggregate them in a certain way.



#---------------------------------------------------------------------------------------------

#----------------------------------------------

## Dahboard
## Regession models

pag6_layout = html.Div([
              html.H3('Regression Model')
             
])




#------------------------------------------------------------------------------------- 
## Clustering
 

## Clean and Prepare Data
rdt_clean = ddf                                                            # From the 'Date' is taken out the 'week of the day' feature, Monday is 0, so the week goes from 0 to 6.
rdt_clean['Day Date']=rdt_clean.index.date                                                                       # From the 'Date' is taken out the 'day date' feature.
rdt_clean = rdt_clean.set_index ('Day Date', drop = True)                                                        # Set 'day date' as index.
cluster_data=rdt_clean.drop(columns=['Holiday', 'RH', 'Solar_Rad[W/m^2]','Months','Power-1'])                                            # Drop useless information.


## K means algorithm :
    # Define the number of clusters and choose the center of each one.
    # Calculate the distance of each of the N points to the center of each cluster.
    # It is a Iterative approach.

model =KMeans(n_clusters=3).fit(cluster_data)
pred = model.labels_
cluster_data['Cluster']=pred
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
score = [kmeans[i].fit(cluster_data).score(cluster_data) for i in range(len(kmeans))]                           # the 'score' indicates the mean distance between the points and the cluster center.                      


## Results:
    # In the Elbow method the optimal number of clusters is 3.


## Dahboard

pag7_layout = html.Div([
              html.H3('Clustering',
              style={'color':'white', 'text-align':'center'}),
              html.H6('With 3 clutsers is possible to distinguish 3 profiles,'
                      'the highest power consumption is between late morning and early afternoon'),
              html.P('Select the Clustering graph:'),
              dcc.Dropdown(
                  id ='dropd-clust',
                  options=[
                      {'label': 'Power vs Temperature', 'value':11},
                      {'label': 'Power vs Week day', 'value':22},
                      {'label': 'Power vs  Hour', 'value':33},
                      {'label': 'Cluster data in 3d', 'value':44},
                      {'label': ' Consumption Pattern', 'value':55},
                      ],
                  value=11
                  ),
              html.Div (id='clust'),
              ])

@app.callback(Output('clust', 'children'),
              Input('dropd-clust', 'value'))
def render_clust (cluster):
                      
                      
    if cluster==11:
        return html.Div([
              dcc.Graph(
                  figure= px.scatter(cluster_data, x='Power[kW]',y='Temperature[C]',
                                     color='Cluster', title = 'Power vs Temeperature')
                  ),])
              
                       
    if cluster==22:
        return html.Div([   
              dcc.Graph(
                  figure= px.scatter(cluster_data, x='Power[kW]',y='Hour',
                                     color='Cluster', title = 'Power vs Hour')
                  ),])
              
                       
    if cluster==33:
        return html.Div([       
              dcc.Graph(
                  figure= px.scatter(cluster_data, x='Power[kW]',y='Week Day',
                                     color='Cluster', title = 'Power vs Week day')
                  ),])
              
                      
    if cluster==44:
        return html.Div([   
                dcc.Graph(
                  figure= px.scatter_3d(cluster_data,x='Hour', z='Power[kW]',y='Week Day',
                                     color='Cluster',height=650, title= 'Cluster data in 3d')
                  ),
              
              
])

                    
    if cluster==55:
        return html.Div([
        html.Center(html.Img(src='data:image/png;base64,{}'.format(encoded_image_dailyp.decode()))),
])



## Result of plots:
    # With 3 clutsers is possible to distinguish 3 profiles.
    # The power consumption is high at low and high temperature (electrical heating and cooling).
    # The high consumption occurs in the hours from 10 to 15.
    # During the weekend the power consumption is low.
    # The highest power consumption is between late morning and early afertnoon.
    # The gap in the patterns is regarding the 1 hour shift between winter and summer. In winter the curve start to increase earlier and reduce earlier than summer.
    # The blue line represent the days in which the people stay in the house most of time.
    # The red line gives the information that the people stay in the house, but they go to work certain times of the day.
    # The green line shows a profile of almost null consumption, this happen when the people are not in the house.







# -------------------------------------------------------------------------------------

# Callbacks for lateral Menu
   
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/":
        return pag1_layout
    elif pathname == "/page-1":
        return pag2_layout
    elif pathname == "/page-2":
        return pag3_layout
    elif pathname == "/page-3":
        return pag4_layout
    elif pathname == "/page-4":
        return pag5_layout
    elif pathname == "/page-5":
        return pag6_layout
    elif pathname == "/page-6":
        return pag7_layout





if __name__ == '__main__':
    app.run_server(debug=False)
