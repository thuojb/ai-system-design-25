# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
from dash import Dash, dcc, html, Input, Output, State
from dash import Dash, dash_table

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

col_style = {'display':'grid', 'grid-auto-flow': 'row'}
row_style = {'display':'grid', 'grid-auto-flow': 'column'}

import plotly.express as px
import pandas as pd

import requests

app = Dash(__name__)

df = pd.read_csv("iris_extended_encoded.csv",sep=',')
df_csv = df.to_csv(index=False)

app.layout = html.Div(children=[
    html.H1(children='Iris classifier'),
    dcc.Tabs([
    dcc.Tab(label="Explore Iris training data", style=tab_style, selected_style=tab_selected_style, children=[

    html.Div([
        html.Div([
            html.Label(['File name to Load for training or testing'], style={'font-weight': 'bold'}),
            dcc.Input(id='file-for-train', type='text', style={'width':'100px'}),
            html.Div([
                html.Button('Load', id='load-val', style={"width":"60px", "height":"30px"}),
                html.Div(id='load-response', children='Click to load')
            ], style=col_style)
        ], style=col_style),

        html.Div([
            html.Button('Upload', id='upload-val', style={"width":"60px", "height":"30px"}),
            html.Div(id='upload-response', children='Click to upload')
        ], style=col_style| {'margin-top':'20px'})

    ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),


html.Div([
    html.Div([
        html.Div([
            html.Label(['Feature'], style={'font-weight': 'bold'}),
            dcc.Dropdown(
                df.columns[:-1].tolist(), # Numeric columns excluding 'species'
                df.columns[0], #defaulr first numeric column
                #['a','b','c'], #<dropdown values for histogram>
                #'a',           #<default value for dropdown>
                id='hist-column'
            )
            ], style=col_style ),
        dcc.Graph( id='selected_hist' )
    ], style=col_style | {'height':'400px', 'width':'400px'}),

    html.Div([

    html.Div([

    html.Div([
        html.Label(['X-Axis'], style={'font-weight': 'bold'}),
        dcc.Dropdown(
            df.columns[:-1].tolist(),
            df.columns[0],
            id='xaxis-column'
            )
        ]),

    html.Div([
        html.Label(['Y-Axis'], style={'font-weight': 'bold'}),
        dcc.Dropdown(
            df.columns[:-1].tolist(),
            df.columns[0],
            id='yaxis-column'
            )
        ])
    ], style=row_style | {'margin-left':'50px', 'margin-right': '50px'}),

    dcc.Graph(id='indicator-graphic')
    ], style=col_style)
], style=row_style),

    html.Div(id='tablecontainer', children=[
        dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], page_size=15,
            id='datatable' )
        ])
    ]),
    dcc.Tab(label="Build model and perform training", id="train-tab", style=tab_style, selected_style=tab_selected_style, children=[
        html.Div([
            html.Div([
                html.Label(['Enter a dataset ID to use in training'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='dataset-for-train', type='text'))
            ], style=col_style | {'margin-top':'20px'}),
            
            html.Div([
                html.Button('New model', id='build-val', style={'width':'90px', "height":"30px"}),
                html.Div(id='build-response', children='Click to build new model and train')
            ], style=col_style | {'margin-top':'20px'}),
            
            html.Div([
                html.Label(['Enter a model ID for re-training'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='model-for-train', type='text'))
            ], style=col_style | {'margin-top':'20px'}),

            html.Div([
                html.Button('Re-Train', id='train-val', style={"width":"90px", "height":"30px"})
            ], style=col_style | {'margin-top':'20px', 'width':'90px'})

        ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),

        html.Div(id='container-button-train', children='')
    ]),
    dcc.Tab(label="Score model", id="score-tab", style=tab_style, selected_style=tab_selected_style, children=[
        html.Div([
            html.Div([
                html.Label(['Enter a row text (CSV) to use in scoring'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='row-for-score', type='text', style={'width':'300px'}))
            ], style=col_style | {'margin-top':'20px'}),
            html.Div([
                html.Label(['Enter a model ID for scoring'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='model-for-score', type='text'))
            ], style=col_style | {'margin-top':'20px'}),            
            html.Div([
                html.Button('Score', id='score-val', style={'width':'90px', "height":"30px"}),
                html.Div(id='score-response', children='Click to score')
            ], style=col_style | {'margin-top':'20px'})
        ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),
        
        html.Div(id='container-button-score', children='')
    ]),

    dcc.Tab(label="Test Iris data", style=tab_style, selected_style=tab_selected_style, children=[
        html.Div([
            html.Div([
                html.Label(['Enter a dataset ID to use in testing'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='dataset-for-test', type='text'))
            ], style=col_style | {'margin-top':'20px'}),
            html.Div([
                html.Label(['Enter a model ID to use in testing'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='model-for-test', type='text'))
            ], style=col_style | {'margin-top':'20px'}),

            html.Div([
                html.Button('Test', id='test-val'),
            ], style=col_style | {'margin-top':'20px', 'width':'90px'})

        ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),

        html.Div(id='container-button-test', children='')
    ])

    ])
])


# STARTING OF MULTI_LINE COMMENT FIELD...move code below above triple quotes to fill in and run

# callbacks for Explore data tab

# callbacks for Explore data tab

# callbacks for Explore data tab

@app.callback(
    Output('load-response', 'children'),
    Input('load-val', 'n_clicks'),
    State('file-for-train', 'value')
)
def update_output_load(n_clicks, filename):
    global df, df_csv
    if n_clicks is not None and filename:
        try:
            df = pd.read_csv(filename, sep=',')
            df_csv = df.to_csv(index=False)
            return 'Load done.'
        except Exception as e:
            return f'Error loading file: {str(e)}'
    return 'Click to load'

@app.callback(
    Output('upload-response', 'children'),
    Input('upload-val', 'n_clicks')
)
def update_output_upload(n_clicks):
    global df_csv
    if n_clicks is not None:
        try:
            files = {'train': ('iris_extended_encoded.csv', df_csv, 'text/csv')}
            response = requests.post('http://localhost:5000/iris/datasets', files=files)
            if response.status_code == 200:
                dataset_id = response.text
                return f'Dataset uploaded, ID: {dataset_id}'
            else:
                return f'Upload failed: {response.text}'
        except Exception as e:
            return f'Upload error: {str(e)}'
    return 'Click to upload'

@app.callback(
    Output('selected_hist', 'figure'),
    Input('hist-column', 'value'),
    Input('load-response', 'children')
)
def update_hist(hist_column_name, load_response):
    if hist_column_name:
        fig = px.histogram(df, x=hist_column_name)
        fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
        fig.update_xaxes(title=hist_column_name)
        return fig
    return {}

@app.callback(
    Output('indicator-graphic', 'figure'),
    Input('xaxis-column', 'value'),
    Input('yaxis-column', 'value'),
    Input('load-response', 'children')
)
def update_graph(xaxis_column_name, yaxis_column_name, load_response):
    if xaxis_column_name and yaxis_column_name:
        fig = px.scatter(x=df[xaxis_column_name].values, y=df[yaxis_column_name].values)
        fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
        fig.update_xaxes(title=xaxis_column_name)
        fig.update_yaxes(title=yaxis_column_name)
        return fig
    return {}

@app.callback(
    Output('tablecontainer', 'children'),
    Input('load-response', 'children')
)
def update_table(load_response):
    return dash_table.DataTable(
        df.to_dict('records'),
        [{"name": i, "id": i} for i in df.columns],
        page_size=15,
        id='datatable'
    )

# callbacks for Training tab

@app.callback(
    Output('build-response', 'children'),
    Input('build-val', 'n_clicks'),
    State('dataset-for-train', 'value')
)
def update_output_build(n_clicks, dataset_id):
    if n_clicks is not None and dataset_id:
        try:
            data = {'dataset': dataset_id}
            response = requests.post('http://localhost:5000/iris/model', data=data)
            if response.status_code == 200:
                model_id = response.text
                return f'Model built and trained, ID: {model_id}'
            else:
                return f'Build failed: {response.text}'
        except Exception as e:
            return f'Build error: {str(e)}'
    return 'Click to build new model and train'

@app.callback(
    Output('container-button-train', 'children'),
    Input('train-val', 'n_clicks'),
    State('model-for-train', 'value'),
    State('dataset-for-train', 'value')
)
def update_output_train(n_clicks, model_id, dataset_id):
    if n_clicks is not None and model_id and dataset_id:
        try:
            response = requests.put(f'http://localhost:5000/iris/model/{model_id}', params={'dataset': dataset_id})
            if response.status_code == 200:
                # Backend returns history as string, parse it or display raw
                history_str = response.text
                return f'Training history: {history_str}'
            else:
                return f'Training failed: {response.text}'
        except Exception as e:
            return f'Training error: {str(e)}'
    return ""

# callbacks for Scoring tab

@app.callback(
    Output('score-response', 'children'),
    Input('score-val', 'n_clicks'),
    State('row-for-score', 'value'),
    State('model-for-score', 'value')
)
def update_output_score(n_clicks, row_data, model_id):
    if n_clicks is not None and row_data and model_id:
        try:
            # Parse CSV row to list of floats
            fields = [float(x) for x in row_data.split(',')]
            response = requests.get(f'http://localhost:5000/iris/model/{model_id}/score', params={'fields': ','.join(map(str, fields))})
            if response.status_code == 200:
                score_result = response.text
                return f'Score result: {score_result}'
            else:
                return f'Scoring failed: {response.text}'
        except Exception as e:
            return f'Scoring error: {str(e)}'
    return 'Click to score'

# callbacks for Testing tab

@app.callback(
    Output('container-button-test', 'children'),
    Input('test-val', 'n_clicks'),
    State('dataset-for-test', 'value'),
    State('model-for-test', 'value')
)
def update_output_test(n_clicks, dataset_id, model_id):
    if n_clicks is not None and dataset_id and model_id:
        try:
            response = requests.post(f'http://localhost:5000/iris/model/{model_id}/test', data={'dataset': dataset_id})
            if response.status_code == 200:
                metrics = response.json()
                fig = px.bar(
                    x=['Accuracy', 'Loss'],
                    y=[metrics['accuracy'], metrics['loss']],
                    title='Test Metrics'
                )
                return dcc.Graph(figure=fig)
            else:
                return f'Testing failed: {response.text}'
        except Exception as e:
            return f'Testing error: {str(e)}'
    return ""

if __name__ == '__main__':
    #app.run_server(debug=True)
    app.run(debug=True)