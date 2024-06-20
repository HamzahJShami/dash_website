import pandas as pd
pd.options.mode.chained_assignment = None
import dash_cytoscape as cyto
import dash
from dash import html, dcc
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.contingency import association
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from graph_functions import Data_pre_processing, nodes_making, edge_making, create_graph, create_analysis,cytograph

diabete_filter_data = Data_pre_processing(2)
nodes = nodes_making(diabete_filter_data)
edge_dataframe = edge_making(diabete_filter_data)
G = create_graph(nodes,edge_dataframe)
analysis_data = create_analysis(G)
elements = cytograph(nodes, edge_dataframe,G,analysis_data)

default_stylesheet = [
    {
        'selector':'node',
        'style': {
            'width': '80px',
            'height':'80px',
            'content': 'data(label)',
            'font-size':'96px',
            'color':'white',
            'background-color': '#FFDF00'
                }
    },
    {
        'selector':'edge',
        'style': {
            'line-color':'dark grey'
        }

    }


]
cytoscape_graph = html.Div([cyto.Cytoscape(id = 'cytograph',
                    className= 'net_obj',
                    elements=elements,
                    responsive=False,
                    style={'width':'100%','height':'100%'},
                    layout={'name': 'dagre',
                        'padding': 1,
                        'nodeRepulsion': '10',
                        'gravityRange': '1.0',
                        'nestingFactor': '1',
                        'edgeElasticity': '10',
                        'idealEdgeLength': '10',
                        'nodeDimensionsIncludeLabels': 'true',
                         'avoidOverlap': 'true',
                        'numIter': '100'
                        
                            },
                    stylesheet=default_stylesheet
                   )],style={ 'height':'100%', 'width':'100%'}, className= 'opacity-15',)

analysis_cards = 'border-white border-3 border border rounded m-3 bg-light'
card_style = {'textAlign':'center','height':'20%', 'width':'80%'}
node_centrality = dcc.Markdown('''Centrality is a very important tool to measure how important a node is in a graph.
                                Here it would be how closely linked to the other diabetes indicators a indicator is.
                                Below cards show Three different measure of centrality, with a brief explanation on how they are calculated''')
degree_centrality = dcc.Markdown('''
                                 Degree calculation of centrality the sum of the number of edges connected to the node. 
                                 This is a simple calculation that can be used to identify the most well connected node. 
                                 In this graph a high degree represents a health indicator that is correlated with a lot 
                                 of diferent health indicators''')
between_centrality = dcc.Markdown('''
                                 Betweeness Centrality measures centrality by summing the amount of times the nodes 
                                 appears in the shortes path between all pairs of nodes in a grpah. The shortest path between a two nodes, 
                                 is the minimum numbers of nodes a path has to go through before you reach the second node.
                                 In this example, a high betweeness shows helath indicator as an important bridge between other health indicators''')
eigen_centrality = dcc.Markdown('''Eigenvector Centrality measures centrality by summing the amount of times a node connects to other important nodes.
                                 The shortest path between a two nodes, 
                                 is the minimum numbers of nodes a path has to go through before you reach the second node.
                                 In this example, a high betweeness shows helath indicator as an important bridge between other health indicators''')


cyto.load_extra_layouts()
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.CYBORG]
                )
server = app.server


app.layout = html.Div(children = [html.Div(children = [html.H1(children='Network Analysis on Diabetes Indicators', 
                                  style={'textAlign':'center','height':'20%'}, className="navbar-expand-lg bg-light "),
 #                        html.Div(children='NavBar', style={'textAlign':'center','height':'2%'}, className="nav opacity-75 border-bottom border-light border-2"),
                         ]),
    html.Div(children = [
        html.Div(children = [
            dcc.Markdown('''
                Introduction
                ---
                This dashboard is a portfolio project built by Hamzah Shami. I used a combination of Ploty's Dash,
                          Networkx, Dash Bootstrap Components, with help from the lovely people suppling the health science 
                and analytics course from University of Exeter. 
                
                My aim for the dashboard is to take the Diabetes Health Indicator set from Kaggle and have fun with it. 
                         I want to show off some data analysis and visualization with Networkx, hopefully providing a interesting insight or, at least, 
                         a new perspective on an old data set. 
  
                To that end the Graph in the center of the screen represents the relationship between a few of the metrics within the dataset. 
                         The nodes represents the indicator, the edge (or line) between the nodes represents the strength of correlation, calculated 
                         with the Pearson correlation coefficient. Some indicators are removed if their correlation with the other indicators fall
                          below a certain threshold (here I use 0.1.) The closer the nodes are too each other the strong the relationship. 
                
                The graph can be toggled to show the relationship based on the different diabetic states of the individuals in the dataset. 
                         Those with Diabetes, Pre-Diabetes condition or with no diabetes. If you hover over a node a series of graph analysis 
                         indicators will be present on the right hand side of the screen, alongside a brief explanation of what they mean
            
                If you want to hire me to create visualise your beautiful data in bespoke dashboard please email: Hamzah.jas@gmail.com.        
                         ''')], style= {'width':'30%'}, className= 'm-3 text-body'),
                 
                           html.Div([cytoscape_graph,
                               dcc.Dropdown(
                               options=[
                                   {'label':'Diabetes','value':2},
                                   {'label':'pre Diabetes','value':1},
                                   {'label':'non Diabetes','value':0}],
                                   value=(2),
                                   className = 'select_box',
                                   id = 'input_graph',
                                   clearable=False,
                                   )
                           ],style={ 'height':'600px', 'width':'40%', 'display':'inline-block'}),
                           
        html.Div(children =[html.Div(node_centrality)], style = {'width':'30%'}, className= 'm-3 text-body text-wrap',id = 'right_analytics')
        
    ],style={'display': 'flex', 'flexDirection': 'row'})
])

@app.callback(
        Output('cytograph','elements'),
        [Input('input_graph','value')]
         
)
def change_graph(value):
    diabete_filter_data = Data_pre_processing(int(value))
    nodes = nodes_making(diabete_filter_data)
    edge_dataframe = edge_making(diabete_filter_data)
    G = create_graph(nodes,edge_dataframe)
    analysis_data = create_analysis(G)
    elements = cytograph(nodes, edge_dataframe,G,analysis_data)
    return elements


@app.callback(
    Output('right_analytics','children'),
    [Input('cytograph', 'mouseoverNodeData')],
    prevent_initial_call=True
)
def display_mouse_over(mouseoverNodeData):
    return [html.Div(children=node_centrality), html.Div(children = ''' Currently Node Selected: '''+ str(mouseoverNodeData['label'])),
            html.Div(children = [html.Div(['''The''', 
                                           html.Span(' Eigenvector',id='eigen_tooltip',
                                                    style={"textDecoration": "underline", "cursor": "pointer"}),
                                          ''' centrality of the node is: ''',
                                            dbc.Tooltip(
                                                    eigen_centrality,
                                                    target="eigen_tooltip"),
                                            str(mouseoverNodeData['centrality'])])],
                    className= analysis_cards ,style=card_style),
            html.Div(children = [html.Div(['''The''', 
                                           html.Span(' Degree',id='degree_tooltip',
                                                    style={"textDecoration": "underline", "cursor": "pointer"}),
                                          ''' centrality of the node is: ''',
                                            dbc.Tooltip(
                                                    degree_centrality,
                                                    target="degree_tooltip"),
                                            str(mouseoverNodeData['degree'])])],
                    className= analysis_cards ,style=card_style),
            html.Div(children = [html.Div(['''The''', 
                                           html.Span(' Betweenness',id='betweenness_tooltip',
                                                    style={"textDecoration": "underline", "cursor": "pointer"}),
                                          ''' centrality of the node is: ''',
                                            dbc.Tooltip(
                                                    between_centrality,
                                                    target="betweenness_tooltip"),
                                            str(mouseoverNodeData['betweenness'])])],
                    className= analysis_cards ,style=card_style)]


if __name__ == "__main__":
    app.run(debug=False,host = '0.0.0.0')