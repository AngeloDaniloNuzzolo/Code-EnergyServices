import dash
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = ['mystyle.css']
df = pd.read_csv('data_class15.csv')


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='IST Energy Monitor - Dashboard 1'),

    html.Div(children='''
        Visualization of total electricity consumption at IST over the last years
    '''),

    dcc.Graph(
        id='yearly-data',
        figure={
            'data': [
                {'x': df.year, 'y': df.Total, 'type': 'bar', 'name': 'Total'},
                {'x': df.year, 'y': df.Civil, 'type': 'bar', 'name': 'Civil'},
                {'x': df.year, 'y': df.Central, 'type': 'bar', 'name': 'Central'},
                {'x': df.year, 'y': df.NorthTower, 'type': 'bar', 'name': 'North Tower'},
                {'x': df.year, 'y': df.SouthTower, 'type': 'bar', 'name': 'South Tower'},
            ],
            'layout': {
                'title': 'IST yearly electricity consumption (MWh)'
            }
        }
    ),

])

if __name__ == '__main__':
    app.run_server(debug=True)