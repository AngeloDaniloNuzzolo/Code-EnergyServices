# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:18:26 2021

@author: danilo
"""
# Import libraries
import pandas as pd                           # Allows reading, writing and handling data.
from matplotlib import pyplot as plt          # Visualization methods, from matplotlib import pyplot as plt.
import seaborn as sb                          # Statistics data visualization based in matplotlib.
import matplotlib.ticker as ticker            # Data visualization with ticker funtion.

# Import libraries for Dashboard
import dash  
from dash.dependencies import Input, Output
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import flask
import glob
import os



# Import Data already cleaned
raw_data_tot= pd.read_csv('IST_Central_Pav_Clean0.csv') #Import data

df= raw_data_tot
df = df.set_index(pd.to_datetime(df['Date'])) 
df['Hour'] = df.index.hour 
print(raw_data_tot.columns)

# Dashboard implementation
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Application
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
image_directory ='/Users/danilo/anaconda3/envs/dashboards/Dashboards/assets/Regression_image/'
images_list = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory))]
static_image_route = '/static/'

colors = {
    'background': '#488A99',
    'text': '#7FDBFF'}
# Organization of dashboard in tabs
app.layout = html.Div(
    style={'backgroundColor' : colors['background']},children =[
    (html.Img(src= 'https://www.eduopinions.com/wp-content/uploads/2018/05/InstitutoSuperiorTecnico-logo.png',
    style = {'height':'7%', 'width':'7%' })),
    html.H1('Central building consumption forecast ',
    style={'text-align': 'center', 'color':'white'}),
    html.H6('Angelo Danilo Nuzzolo-ist1100830',
    style={'color':'lightgrey'}),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw data', value='tab-1'),
        dcc.Tab(label='Exploratory data analysis 1', value='tab-2'),
        dcc.Tab(label='Exploratory data analysis 2', value='tab-3'),
        dcc.Tab(label='Feature selection', value='tab-4'),
        dcc.Tab(label='Regression models', value='tab-5'),
        dcc.Tab(label='Clustering', value='tab-6'),
    ]),
    html.Div(id='tabs-content')
])

##  RAW DATA VISUALIZATION : Interactive Table 
tab1_layout = html.Div([
               html.H3('Data cleaned table',
               style={'color':'white'}),
               dash_table.DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": i, "id": i, "deletable": True, "selectable": True, "hideable": True}
            
            for i in df.columns
        ],
        data=df.to_dict('records'),  # the contents of the table        
        filter_action="none",       # allow filtering of data by user ('native') or not ('none')
        sort_action="native",       # enables data to be sorted per-column by user or not ('none')
        sort_mode="single",         # sort across 'multi' or 'single' columns
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

#----------------------------------------------------------------------


# EXPLORATORY DATA ANALYSIS (Raw data cleaned)
    # The EDA is a philosophy to explore data (mostly with visual representation) and propose new hypothesis.

print (raw_data_tot.describe())                                                # quick statistical check
##Comments
     # 1.: The minimum and maximum values of the features appear reasonable;
     # 2.: Holiday=1, no Holiday=0, it can mean calendar holiday or summer holiday;


## Calculate basic statistic : Correlations
correlation=raw_data_tot.corr('spearman')
print (correlation)
## Correlation:
     #1.: Spearman correlation = 1  indicates a perfect association between ranks;
     #2.: Spearman correlation = -1  indicates a perfect negative association between ranks;
     #3.: Spearman correlation = 0  indicates no association between ranks.

## Comments : 
    # It can be noticed a positive correlation between temperature and power, this means that an increasing of temperature leads to an increase in power consumption (air conditioners or fans are turned on).
    # In the case of Solar Radiation it is possible to see a important positive correlation, this means that the trade off between the possibility to turning the light off (saving energy) and the increasing of temperature is won by the latter.
    # The negative correlation between RH and power is explained by the necessity to humidify the air with an air conditioner when the RH drops below 40%.
    # The correlation between Holiday and Power is negative beacuse people are generally not at home during the Holidays.


## Power
fig, ax1 = plt.subplots()                                                      # create objects of the plot.
fig.set_size_inches (20,10)                                                    # define figure size.

ax1.xaxis.set_major_locator (ticker.MultipleLocator(1000))                      # define the interval between ticks on x axis
ax1.xaxis.set_tick_params (which= 'major', pad = 8, labelrotation = 50)        # parameter of major labels of x axis : pad =distance to the axis.
                                                                               # Label rotation = angle of label text (in degrees)
#plot
plt.plot (raw_data_tot ['Power[kW]'], '-o', color = 'blue',                    # x axis labels -  symbol type.
          markersize = 12, linewidth = 1,                                      # point size - line thickness.
          markerfacecolor = 'red',                                             # color inside the points
          markeredgecolor = 'black',                                           # color of edge
          markeredgewidth= 3)
plt.ylabel('Power[kW]')
plt.xlabel('Date')


##Notes:
  # As a result of lack of data (data missing at the end of 2018), an interpolation of then weather data was made which however did not lead to any significant result, for this reason the interpolation code was left as comment.
  # This beacuse regarding the weather data is not a good choice copy or make interpolation of them,  it is better gathering data from another weather station.

# #Interpolation 
# raw_data_tot_inter=raw_data_tot.resample('D').mean()
# raw_data_tot_inter['Temperature[C]'] = raw_data_tot_inter['Temperature[C]'].interpolate()    


## Temperature          
fig2, ax = plt.subplots() 
fig2.set_size_inches (20,10)                                                   # define figure size.

ax.xaxis.set_major_locator (ticker.MultipleLocator(1000))                       # define the interval between ticks on x axis
ax.xaxis.set_tick_params (which= 'major', pad = 8, labelrotation = 50)        
plt.plot (raw_data_tot ['Temperature[C]'], '-*', color = 'blue',               # x axis labels-data-symbol type.
          markersize = 20, linewidth = 0.5,                                    # point size-line thickness.
          markerfacecolor = 'red',                                             # color inside the points
          markeredgecolor = 'orange',                                          # color of edge
          markeredgewidth= 1.5)
plt.ylabel('Temperature[C]')
plt.xlabel('Date')

          
## Relative Humidity
fig3, ax = plt.subplots() 
fig3.set_size_inches (20,10)                                                   # define figure size

ax.xaxis.set_major_locator (ticker.MultipleLocator(1000))                       # define the interval between ticks on x axis
ax.xaxis.set_tick_params (which= 'major', pad = 8, labelrotation = 50)        
plt.plot (raw_data_tot ['Relative_Humidity'], '-p', color = 'blue',            # x axis labels- data-symbol type
          markerfacecolor = 'red',                                             # color inside the points
          markeredgecolor = 'purple',                                          # color of edge
          markeredgewidth= 1.5)
plt.ylabel('Relative_Humidity')
plt.xlabel('Date')

## Solar radiation
fig4, ax = plt.subplots() 
fig4.set_size_inches (20,10)                                                   # define figure size

ax.xaxis.set_major_locator (ticker.MultipleLocator(1000))                       # define the interval between ticks on x axis
ax.xaxis.set_tick_params (which= 'major', pad = 8, labelrotation = 50)        
plt.plot (raw_data_tot ['Solar_Radiation[W/m^2]'], '-p', color = 'blue',       # x axis labels- data-symbol type
          markerfacecolor = 'red',                                             # color inside the points
          markeredgecolor = 'purple',                                          # color of edge
          markeredgewidth= 1.5)
plt.ylabel('Solar_Radiation[W/m^2]')
plt.xlabel('Date')

## Overall Results:
    # The time series plots have in common the lack of the data at the end of 2018 (this means that the model will not be able to foresee in accurate way this specific period).
    # The power trend varies almost costantly between a minimum (90 kW) and maximum value (450 kW) in the whole year with peaks during the summer months and with some missing consumption areas (it might be when the people are not present in the house).
    # Power consumption points below the minimum value (90 kW) will be considered as outliers, since the occurence of these point is very small, they can therefore be considered as errors or occasional values.
    # Regarding the temeperature and solar radiation trends, they look very similar, with mamimums in summer months and minimums in winter months, that points out the positive correlation with Power consumption.
    # As regards relative humidity, it seems to have an opposite trend compared to other features, that's explain the small negative correlation with Power consumption.


   
## Correlation plots:
    # To confirm what has been said prevoiusly, correlations between features are analyzed with particular attention to power.
  
## Reset the index to make the plot
raw_data_tot=raw_data_tot.reset_index()                                      

## Power vs Temperature
x= (raw_data_tot.iloc[1000:1250]['Date'])                                      # zoom of the feature 'Date'
y1= (raw_data_tot.iloc[1000:1250]['Power[kW]'])                                # zoom of the feature 'Power'
y2= (raw_data_tot.iloc[1000:1250]['Temperature[C]'])                           # zoom of the feature 'temperature'
fig5, ax1 = plt.subplots()
fig5.suptitle('Power vs Temperature')
ax2 =ax1.twinx()
curve1 =ax1.plot(x, y1, label ='Power', color='blue')
curve2 =ax2.plot(x, y2, label='Temp', color='red')
ax1.set_ylabel('Power[kW]',color='blue')
ax2.set_ylabel('Temperature[C]', color='red')
plt.show()

## Power vs Relative Humidity
x= (raw_data_tot.iloc[1000:1250]['Date'])                                      # zoom of the feature 'Date'
y1= (raw_data_tot.iloc[1000:1250]['Power[kW]'])                                # zoom of the feature 'Power'
y2= (raw_data_tot.iloc[1000:1250]['Relative_Humidity'])                        # zoom of the feature 'Relative humidity'
fig11, ax1 = plt.subplots()
fig11.suptitle('Power vs Relative humidity')
ax2 =ax1.twinx()
curve1 =ax1.plot(x, y1, label ='Power', color='blue')
curve2 =ax2.plot(x, y2, label='RH', color='purple')
ax1.set_ylabel('Power[kW]',color='blue')
ax2.set_ylabel('Relative_Humidity', color='purple')
plt.show()

## Power vs Solar Radiation
x= (raw_data_tot.iloc[1000:1250]['Date'])                                      # zoom of the feature 'Date'
y1= (raw_data_tot.iloc[1000:1250]['Power[kW]'])                                # zoom of the feature 'Power'
y2= (raw_data_tot.iloc[1000:1250]['Solar_Radiation[W/m^2]'])                   # zoom of the feature 'Solar Radiation'
fig6, ax1 = plt.subplots()
fig6.suptitle('Power vs Solar Radiation')
ax2 =ax1.twinx()
curve1 =ax1.plot(x, y1, label ='Power[kW]', color='blue')
curve2 =ax2.plot(x, y2, label='Power[kW]', color='orange')
ax1.set_ylabel('Power[kW',color='blue')
ax2.set_ylabel('Solar_Radiation[W/m^2]', color='orange')
plt.show()

## Scatter plots:
    # A scatter plot uses dots to represent values for two different numeric variables.
    # The position of each dot on the horizontal and vertical axis indicates values for an individual data point.
    # Scatter plots are used to observe relationships between variables.


fig9, axes = plt.subplots(1,figsize= (20,10))
fig9.suptitle('Power vs Temperature and Solar radiation')
sb.scatterplot(data=raw_data_tot, x='Temperature[C]', y='Power[kW]', hue='Holiday', style='Holiday')



# These last plots reinforce what has been said so far:
    # The power consumption follows the temperature and solar radiation trends with a positive correlation (except for some missing consumption).
    # Regarding the relative humididty, it is slighly negatively correlated with power consumption (for this reason it will not be included on the model)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
## Discover outliers with visualization tools.    

## Boxplot
## Boxplots are a standardized way of displaying the distribution of data based on a five number summary (“minimum”, first quartile (Q1), median, third quartile (Q3), and “maximum”):
    # median (Q2/50th Percentile), it is the middle value of the dataset.
    # first quartile (Q1/25th Percentile), it is the middle number between the smallest number (not the “minimum”) and the median of the dataset.
    # third quartile (Q3/75th Percentile), it is the middle value between the median and the highest value (not the “maximum”) of the dataset.
    # interquartile range (IQR): 25th to the 75th percentile.
    # outliers (shown as black circles).
    # “maximum”, Q3 + 1.5*IQR
    # “minimum”, Q1 -1.5*IQR

##Histogram:
## Histogram provides a visual interpretation of numerical data by showing the number of data points that fall within a specified range of values (called “bins”) where:
    # The X-axis are intervals that show the scale of values which the measurements fall under.
    # The Y-axis shows the number of times that the values occurred within the intervals set by the X-axis.
    # The height of the bar shows the number of times that the values occurred within the interval, while the width of the bar shows the interval that is covered. 
    # it highlight the location of Data, and so the spread of data and the presence of ouliers

## Violin plot:
      # it is similar to box plots, except that they also show the probability density of the data at different values
    
    
## Power    
fig7, axes = plt.subplots(1,3, figsize=(15,5))
fig7.suptitle('Boxplot - Histogram - Violin plot')  
sb.boxplot (ax=axes[0], data=raw_data_tot, x='Power[kW]', color= 'red')                              # Boxplot using seaborn
sb.histplot( ax=axes[1], data=raw_data_tot , x='Power[kW]', bins=100 , color= 'blue' )               # Histogram using seaborn
sb.violinplot(ax=axes[2], x=raw_data_tot['Power[kW]'], inner ='box', color= 'green')                 # Violin plot using seaborn

## Temperature    
fig7, axes = plt.subplots(1,3, figsize=(15,5))
fig7.suptitle('Boxplot - Histogram - Violin plot')  
sb.boxplot (ax=axes[0], data=raw_data_tot, x='Temperature[C]', color= 'red')                         # Boxplot using seaborn
sb.histplot( ax=axes[1], data=raw_data_tot , x='Temperature[C]', bins=100 , color= 'blue' )          # Histogram using seaborn
sb.violinplot(ax=axes[2], x=raw_data_tot['Temperature[C]'], inner ='box', color= 'green')            # Violin plot using seaborn

## Solar radiation    
fig8, axes = plt.subplots(1,3, figsize=(15,5))
fig8.suptitle('Boxplot - Histogram - Violin plot')  
sb.boxplot (ax=axes[0], data=raw_data_tot, x='Solar_Radiation[W/m^2]', color= 'red')                 # Boxplot using seaborn
sb.histplot( ax=axes[1], data=raw_data_tot , x='Solar_Radiation[W/m^2]', bins=20 , color= 'blue' )   # Histogram using seaborn
sb.violinplot(ax=axes[2], x=raw_data_tot['Solar_Radiation[W/m^2]'], inner ='box', color= 'green')    # Violin plot using seaborn

## Result power consumption:
    # The most of the points fall at 300 kW and within the interval between 100-200 kW (with higher peak), therefore most of the time the consumption is low. 
    # The violin and box plots highlight high values and very low values of consumption as outliers, in the cleaning process these points will be removed.
   

## Result temperature and solar radiation:
    # In case of temperature the most of points fall within the interval between 10-20 °C (a cool climate), for solar radiation the number occurences are concentrated at 0 value (during the night the solar radiation is 0).
   

#Comments:
        # It is clearly possible to notice power peaks at low and high temperaure, this allow us to understand that the building is heated and cooled electrically, with greater consumption during the summer months.
        # During the holidays the consumption is low.
        # The 'trade off' between the high consumption during the night due to the switchig lihts on , and the high consumption at high solar radiation during the summer months are again clearly visible.

#----------------------------------------------------------------------------

## Exploratory data analysis -- Dashboard implementation
# Time series graph

tab2_layout =  html.Div(children=[
               html.H3('Time series graph',
                       style={'color':'white'}),
             dcc.Graph(
                id="yearly-data1",
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
                id="yearly-data2",
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
    

   
#Violin plots, to visualize outliers


    html.H3('Violin Plots',
    style={'color':'white'}),
    html.Hr(),
    html.Div(className='two columns', children=[
        dcc.RadioItems(
            id='items',
            options=[
                {'label': 'Power', 'value': 'Power[kW]'},
                {'label': 'Temperature', 'value': 'Temperature[C]'},
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
print (Q1)
Q3 =df['Power[kW]'].quantile(0.75)                                                                              # Third quartile
print (Q3)
IQR = Q3- Q1
print (IQR)                                                                                                               # Inter quartile range

rdt_clean =df[((df['Power[kW]'] > (Q1 - 1.5*IQR)) & (df['Power[kW]'] < (Q3 + 1.5 *IQR)))]   # we accept all the data in the interval
df_sort_kW = rdt_clean.sort_values (by = 'Power[kW]', ascending = False)
print (df_sort_kW)
# Comments :
    # The points outside the interval are deleted.
    
plt.plot(rdt_clean['Power[kW]'])    



# Let's try tosee more in detailed the scatter plots
## Dashboard 

df=rdt_clean
df = df.set_index(pd.to_datetime(df['Date']))                   # make Date into 'datetime' and then index
df = df.drop (columns = 'Date')
df['year'] = df.index.year
df['Months'] = df.index.month  
df['days']  = df.index.day
df=df[df['year']==2017]




mark_values = {1:'January',2:'February',3:'March',4:'April',
               5:'May',6:'June',7:'July',8:'August',
               9:'September',10:'October',11:'November',12:'December'}


tab3_layout = html.Div([
              html.Div([
              html.H2("Scatter plots",
              style={"text-align": "center", "color":"white"}),
              html.H6("The scatter plots below are represented by the most important "
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
                 "left":"10%"})

])

#Callbacks for scatter plots                 
@app.callback(                        
    Output('the_graph1','figure'),
    [Input('slider','value')]
)

def update_graph1(value):
    dff=df[(df['Months']>=value[0])]
   
   
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
       

 
#--------------

# FEATURE SELECTION AND ENGINEERING:
    # It consist in choosing the set of relevant variables that can be used to develop a model.
    # It is used to improve the accuracy, avoid overfitting and reduce the complexity of the model.
    
ddf= rdt_clean
print (ddf)
#df = df.set_index(pd.to_datetime(df['Date']))                   # make Date into 'datetime' and then index
ddf = ddf.drop (columns = 'Date')
ddf['Months'] = ddf.index.month 
# df_data = df_data.drop (columns = 'year')
# df_data = df_data.drop (columns = 'days')
ddf['Week Day'] = ddf.index.dayofweek                                  # Add day of week column



##To calculate the power at a certain time instant, we need to know the power at time t-1
ddf['Power-1']=ddf['Power[kW]'].shift(1)                               # Previous hour consumption
ddf=ddf.dropna()                                                       # Drop NaN values
print(ddf.head())
print(ddf.dtypes)


## The function of feature section doesn't work with data frame but array structure.
## The Power is the output, and the the other features are the input.
## Define input and outputs.
X=ddf.values                                                               # in there i copy the values

Y=X[:,0]                                                                       # the output of the model is the power
X=X[:,[1,2,3,4,5,6,7,8]]                                                       # x is substituted with the array defined by X=df_data.values
print(Y)
print(X)

## Filter Methods (k best):
    # It uses measures to score the data features (Correlation, mutual information,t-test).
    # The top score features are chosen to be part of the model
    # Redundancy in selected features.
    
from sklearn.feature_selection import SelectKBest                              # selection method
from sklearn.feature_selection import f_regression, mutual_info_regression     # score metrix


features=SelectKBest(k=5,score_func=f_regression)                              # Test different k number of features, uses f-test ANOVA
fit=features.fit(X,Y)                                                          # calculates f_regression of the features (calculates the correlation between features and output )
print(fit.scores_)
features_results=fit.transform(X)
print(features_results)                                                        # k=5 : Power-1, Solar Radiation, Week day,Relative Humidity, temperature
plt.bar([i for i in range(len(fit.scores_))],fit.scores_)
## Comments :
    # from there, the highest value obtained for Y gives the value that most affect the forecasting.
    # the output shows the pearson correlation between feature and power. The highest value shows the highest correlation.
    # K is used to understand how many features we want to stamp at the end. k=1 it means that it stamp the most important.


## Mutual information:
features=SelectKBest(k=5,score_func=mutual_info_regression)                    # Test different k number of features, uses mutual information
fit=features.fit(X,Y) 
print(fit.scores_)
features_results=fit.transform(X)
print(features_results)                                                        # k=5 :Power-1, Hour, Solar radiation, Week day, Temperature
plt.bar([i for i in range(len(fit.scores_))],fit.scores_)
featureSelected= [len(fit.scores_)]
## Comments
   # The highest is the output value more is the amount of information that this variable can obtain observing the input ones.
   # The output shows the mutual correlation between feature and power. The highest value shows the highest correlation.
   # K is used to understand how many features we want to stamp at the end. k=1 it means that it stamp the most important.


from sklearn.ensemble import RandomForestRegressor                             # Random Forest Regressor is used

model = RandomForestRegressor()                                                # Verification of chosen features
model.fit(X, Y)
print(model.feature_importances_)                                              # Power-1, Hour, solar radiation, Temperature, Week day
print (ddf)


## Final Result:
    # Power-1, Hour, Solar radiation,Week day, Temperature are the most important features, they will be included in the model.


#--------
# dashboard

# tab4_layout = html.Div([
#     html.Div([ 
#     html.Label('Dropdown'),
#     dcc.Dropdown(
#         id='dropdown',
#         options=[
#             {'label': 'f_regression', 'value': features_results2},
#             {'label': 'mutual_regression', 'value': features_results1}
#         ],
#         value=features_results2,
#         style={'display': 'block'}
#     )
#     ]),


# html.Div(dcc.Graph(id='histo'), className='ten columns')
# ])

# @app.callback(
#     Output('histo', 'children'), [
#     Input('dropdown', 'value')])

# def update_graph5(value):
#     return {
#         'data': [
#             {
#                 'type':'bar',
#                 'x': df[value]
                
#             }
#         ],
#         'layout': {
#             'margin': {'l': 30, 'r': 10, 'b': 30, 't': 0}
#         }
#     }    
        


# features = ['Power[kW]', 'Temperature[C]', 'Relative_Humidity', 'Solar_Radiation[W/m^2]', 'Holiday', 'Hour',
#           'Months', 'Week Day', 'Power-1']

tab4_layout = html.Div(children=[
    html.H1(children='Features selection'),

    html.Div(children='''
        Is examinated with the final score, the correlation between Power and the other features
    '''),

    dcc.Graph(
        id='bar chart1',
        figure={
            'data': [
            {'x': ['Temperature[C]'], 'y': [489.30025279], 'type': 'bar', 'name': 'Temperature[C]'},
            {'x': ['Relative_Humidity'], 'y': [640.97865389], 'type': 'bar', 'name': 'Relative_Humidity'},
            {'x': ['Solar_Radiation[W/m^2]'], 'y': [3344.74010083], 'type': 'bar', 'name': 'Solar_Radiation[W/m^2]'},
            {'x': ['Holiday'], 'y': [157.37249663], 'type': 'bar', 'name': 'Holiday'},
            {'x': ['Hour'], 'y': [257.94354102], 'type': 'bar', 'name': 'Hour'},
            {'x': ['Months'], 'y': [85.87449002], 'type': 'bar', 'name': 'Months'},
            {'x': ['Week Day'], 'y': [1017.49989346], 'type': 'bar', 'name': 'Week Day'},
            {'x': ['Power-1'], 'y': [77226.12283949], 'type': 'bar', 'name': 'Power-1'},
                        
               
           ],
            'layout': {
                'title': 'f-test ANOVA','color': 'lightblue'
            }
        }
    ),
    
     dcc.Graph(
        id='bar chart2',
        figure={
            'data': [
            {'x': ['Temperature[C]'], 'y': [0.11264299], 'type': 'bar', 'name': 'Temperature[C]'},
            {'x': ['Relative_Humidity'], 'y': [0.07286712], 'type': 'bar', 'name': 'Relative_Humidity'},
            {'x': ['Solar_Radiation[W/m^2]'], 'y': [0.28516043], 'type': 'bar', 'name': 'Solar_Radiation[W/m^2]'},
            {'x': ['Holiday'], 'y': [0.0216017], 'type': 'bar', 'name': 'Holiday'},
            {'x': ['Hour'], 'y': [0.52246417], 'type': 'bar', 'name': 'Hour'},
            {'x': ['Months'], 'y': [0.10762179], 'type': 'bar', 'name': 'Months'},
            {'x': ['Week Day'], 'y': [0.17243253], 'type': 'bar', 'name': 'Week Day'},
            {'x': ['Power-1'], 'y': [1.43884488], 'type': 'bar', 'name': 'Power-1'},
                        
            ],
            'layout': {
                'title': 'Mutual information', 'color': 'lightblue'
            }
        }
    ),
     
     dcc.Graph(
        id='bar chart3',
        figure={
            'data': [
            {'x': ['Temperature[C]'], 'y': [3.84010066e-03], 'type': 'bar', 'name': 'Temperature[C]'},
            {'x': ['Relative_Humidity'], 'y': [2.58759955e-03], 'type': 'bar', 'name': 'Relative_Humidity'},
            {'x': ['Solar_Radiation[W/m^2]'], 'y': [3.99018106e-03], 'type': 'bar', 'name': 'Solar_Radiation[W/m^2]'},
            {'x': ['Holiday'], 'y': [2.26997734e-04], 'type': 'bar', 'name': 'Holiday'},
            {'x': ['Hour'], 'y': [9.52311771e-02], 'type': 'bar', 'name': 'Hour'},
            {'x': ['Months'], 'y': [2.38661804e-03], 'type': 'bar', 'name': 'Months'},
            {'x': ['Week Day'], 'y': [3.55102266e-03], 'type': 'bar', 'name': 'Week Day'},
            {'x': ['Power-1'], 'y': [8.88186303e-01], 'type': 'bar', 'name': 'Power-1'},
                        
            ],
            'layout': {
                'title': 'Random Forest Regressor', 'color': 'lightblue'
            }
        }
    )
    
    
    
    

])

#----------------------------
##Regression 

tab5_layout = html.Div([html.H4('Regression models'),
    dcc.Dropdown(
        id='image_clust',        
        options=[{'label':i, 'value':i} for i in images_list],
        value=images_list[0]
    ),
    html.Img(id='outimage')
])


@app.callback(
    dash.dependencies.Output('outimage','src'),
    [dash.dependencies.Input('image_clust','value')])

def update_image_src(value):
    return static_image_route + value


@app.server.route('{}<image_path>.png'.format(static_image_route))

def serve_image (image_path):
    image_name = '{}.png'.format(image_path)
    if image_name not in images_list:
        raise Exception ('"{}" is not included'.format(image_path))
    return flask.send_from_directory(image_directory,image_name)



#------------------- 
## Clustering
    # Clustering is the task of dividing the population or data points into a number of groups such that data points in the same groups are more similar to other data points in the same group than those in other groups.
    # The approch used is the point assignment one.
# Comments  :
    # In this section is performed an unsupervised classification technique in the way to group the data points in subesets as function of similarities between them.
    # The aims of this section are to indentify consumption patterns (profiles), and to discover some usefull feature that will improve the final model.

from sklearn.cluster import KMeans

## Clean and Prepare Data
rdt_clean = ddf                                                            # From the 'Date' is taken out the 'week of the day' feature, Monday is 0, so the week goes from 0 to 6.
rdt_clean['Day Date']=rdt_clean.index.date                                                                       # From the 'Date' is taken out the 'day date' feature.
rdt_clean = rdt_clean.set_index ('Day Date', drop = True)                                                        # Set 'day date' as index.
print(rdt_clean.columns)
cluster_data=rdt_clean.drop(columns=['Holiday', 'Relative_Humidity', 'Solar_Radiation[W/m^2]','Months','Power-1'])                                            # Drop useless information.
print(cluster_data)


## K means algorithm :
    # Define the number of clusters and choose the center of each one.
    # Calculate the distance of each of the N points to the center of each cluster.
    # It is a Iterative approach.

## Elbow method:
    # In the Elbow method increasing the number of clusters will naturally improve the fit since there are more parameters (more clusters) to use, but that at some point there is a over-fitting.
model =KMeans(n_clusters=3).fit(cluster_data)
pred = model.labels_
cluster_data['Cluster']=pred
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
print (kmeans)
score = [kmeans[i].fit(cluster_data).score(cluster_data) for i in range(len(kmeans))]                           # the 'score' indicates the mean distance between the points and the cluster center.                      
print(score)
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()
print(cluster_data)

## Results:
    # In the Elbow method the optimal number of clusters is 3.
  
    
## Cluster analysis    
ax1=cluster_data.plot.scatter(x='Power[kW]',y='Temperature[C]',color=[plt.cm.flag(float(i) /10) for i in cluster_data['Cluster']])       # clustering power vs temperature
plt.title('Power profile clustered with respect to the temperature')
ax2=cluster_data.plot.scatter(x='Power[kW]',y='Hour',color=[plt.cm.flag(float(i) /10) for i in cluster_data['Cluster']])                # clustering power vs hours
plt.title('Power profile clustered with respect to the hours')
ax3=cluster_data.plot.scatter(x='Power[kW]',y='Week Day',color=[plt.cm.flag(float(i) /10) for i in cluster_data['Cluster']])             # clustering power vs week day
plt.title('Power profile clustered with respect to the week day')


fig10 = plt.figure()
ax = plt.axes(projection="3d")

cluster_0=cluster_data[pred==0]
cluster_1=cluster_data[pred==1]
cluster_2=cluster_data[pred==2]


ax.scatter3D(cluster_0['Hour'], cluster_0['Week Day'],cluster_0['Power[kW]'],c='red');
ax.scatter3D(cluster_1['Hour'], cluster_1['Week Day'],cluster_1['Power[kW]'],c='blue');
ax.scatter3D(cluster_2['Hour'], cluster_2['Week Day'],cluster_2['Power[kW]'],c='orange');

plt.show()

## Result of plots:
    # As highlighted before, the power consumption is high at low and high temperature (electrical heating and cooling).
    # The high consumption occurs in the hours from 10 to 15.
    # During the weekend the power consumption is low.


## Dahboard

tab6_layout = html.Div([
              html.H3 ('Clustering'),
              dcc.Dropdown(
                  id ='dropd-clust',
                  options=[
                      {'label': 'Power vs Temperature', 'value':1},
                      {'label': 'Power vs Week day', 'value':2},
                      {'label': 'Power vs  Hour', 'value':3},
                      {'label': 'Cluster data in 3d', 'value':4},
                      ],
                  value=1
                  ),
              html.Div (id='clust'),
              ])

@app.callback(Output('clust', 'children'),
              Input('dropd-clust', 'value'))
def render_clust (cluster):
                      
                      
    if cluster==1:
        return html.Div([
              dcc.Graph(
                  figure= px.scatter(cluster_data, x='Power[kW]',y='Temperature[C]',
                                     color='Cluster', title = 'Power vs Temeperature')
                  ),])
              
                       
    if cluster==2:
        return html.Div([   
              dcc.Graph(
                  id='scatter2',
                  figure= px.scatter(cluster_data, x='Power[kW]',y='Hour',
                                     color='Cluster', title = 'Power vs Hour')
                  ),])
              
                       
    if cluster==3:
        return html.Div([       
              dcc.Graph(
                  figure= px.scatter(cluster_data, x='Power[kW]',y='Week Day',
                                     color='Cluster', title = 'Power vs Week day')
                  ),])
              
                      
    if cluster==4:
        return html.Div([   
                dcc.Graph(
                  figure= px.scatter_3d(cluster_data,x='Hour', z='Power[kW]',y='Week Day',
                                     color='Cluster', height= 800, title= 'Cluster data in 3d')
                  )
              
              
])












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
     elif tab == 'tab-4':
        return tab4_layout
     elif tab == 'tab-5':
        return tab5_layout
     elif tab == 'tab-6':
        return tab6_layout
   



if __name__ == '__main__':
    app.run_server(debug=False)
