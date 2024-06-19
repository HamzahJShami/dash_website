
import pandas as pd
pd.options.mode.chained_assignment = None
import dash_cytoscape as cyto
import dash
from dash import html, dcc
import numpy as np
import plotly.express as px
from scipy.stats.contingency import association
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from sklearn.preprocessing import StandardScaler, MinMaxScaler


#Load in data and clean remove columns I am not overly interested in. This includes some continous variable that I might add back in, but they need to be handed differently (I think)
def Data_pre_processing(Diabetes):
    diabetes_data = pd.read_csv('Data/diabetes_012_health_indicators_BRFSS2015.csv')
    diabete_filter_data = diabetes_data.loc[diabetes_data['Diabetes_012'] == Diabetes]
    diabete_filter_data= diabete_filter_data.drop(['BMI','MentHlth','PhysHlth','Age','Education','Income','Diabetes_012'],axis=1)
    diabete_filter_data['id_patient'] = range(1,len(diabete_filter_data)+1)
    diabete_filter_data.set_index('id_patient')
    diabete_filter_data = diabete_filter_data.reset_index()
    diabete_filter_data=diabete_filter_data.rename(columns = {'HighChol':'High Cholesterol','HighBP':'High Blood Pressure',
                                                               'CholCheck':'Cholesterol Check up','HeartDiseaseorAttack':'Have heart disease or attack',
                                                               'PhysActivity':'Physical Activities','HvyAlcoholConsump':'Heavy Alcohol Consumption',
                                                               'AnyHealthcare':'Have health care coverage','NoDocbcCost':'Avoided seeing a doctor because of cost',
                                                               'GenHlth':'General Health','DiffWalk':'Difficulty Walking','Sex':'Gender'})


    return diabete_filter_data 


#Format the column names from the raw data files into node names for networkx graph
def nodes_making(diabete_filter_data):
    nodes_frame = diabete_filter_data.drop('id_patient',axis=1)
    nodes_raw = pd.DataFrame(nodes_frame.columns.values)
    nodes_raw['id'] = pd.DataFrame(nodes_frame.columns.values)
    nodes = nodes_raw.rename(columns={0:'Label'})
    nodes['Description']=nodes['Label'].replace({'HighChol':'High Cholertol','HighBP':'High Blood Pressure',
                                                               'CholCheck':'Cholesterol Check up','HeartDiseaseorAttack':'Have heart disease or attack',
                                                               'PhysActivity':'Participate in Physical Activities','HvyAlcoholConsump':'Heavy Alcohol Consumption',
                                                               'AnyHealthcare':'Have health care coverage','NoDocbcCost':'Avoided seeing a doctor because of cost',
                                                               'GenHlth':'General Health','DiffWalk':'Difficulty Walking'})
    nodes['Label']=nodes['Label'].replace({'HighChol':'High Cholertol','HighBP':'High Blood Pressure',
                                                               'CholCheck':'Cholesterol Check up','HeartDiseaseorAttack':'Have heart disease or attack',
                                                               'PhysActivity':'Participate in Physical Activities','HvyAlcoholConsump':'Heavy Alcohol Consumption',
                                                               'AnyHealthcare':'Have health care coverage','NoDocbcCost':'Avoided seeing a doctor because of cost',
                                                               'GenHlth':'General Health','DiffWalk':'Difficulty Walking'})

    return nodes


#
def edge_making(diabete_filter_data):
    diabetes_pivot = pd.melt(diabete_filter_data, id_vars= 'id_patient', value_vars=['High Blood Pressure',
    'High Cholesterol',
    'Cholesterol Check up',
    'Smoker',
    'Stroke',
    'Have heart disease or attack',
    'Physical Activities',
    'Fruits',
    'Veggies',
    'Heavy Alcohol Consumption',
    'Have health care coverage',
    'Avoided seeing a doctor because of cost',
    'General Health',
    'Difficulty Walking',
    'Gender',
    'id_patient'])
    diabetes_pivot['variableCat'] = diabetes_pivot['variable'].astype('category')
    diabetes_pivot['variableCatCode'] = diabetes_pivot['variableCat'].cat.codes
    variable_list = max(diabetes_pivot.variableCatCode) +1
    servMove = np.zeros((variable_list, variable_list))
    singles = np.zeros((1))
    clientIDUni = diabetes_pivot.variableCat.unique()
    edge_list = []
    for source in clientIDUni:
        for target in clientIDUni:
            edge_list.append([source,target])

    edge_dataframe = pd.DataFrame(edge_list)
    edge_dataframe = edge_dataframe.rename(columns={0:'Source',1:'Target'})

    edge_dataframe['Weight'] = ""
    edges_matrix = diabete_filter_data.corr(method='pearson')
    edge_list = []
    for source in clientIDUni:
        edge_something = edges_matrix[source]

        for target in clientIDUni:
            weight = abs(edge_something[target])
            sign = 0
            if edge_something[target] < 0:
                sign = -1
            else:
                sign = +1
            edge_list.append([target,source,weight,sign])

    edge_dataframe = pd.DataFrame(edge_list)
    edge_dataframe = edge_dataframe.rename(columns={0:'Source',1:'Target',2:'Weight',3:'Sign'})
    edge_dataframe['Type']='Directed'
    edge_dataframe['id'] = range(1,len(edge_dataframe)+1)
    edge_dataframe = edge_dataframe[edge_dataframe.Weight != 1]
    edge_dataframe = edge_dataframe[edge_dataframe.Weight > 0.1]
    return edge_dataframe
