
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from scipy.stats.contingency import association
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
from sklearn.preprocessing import MinMaxScaler


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


#create the edges i.e. the correlation between the health indicators. 
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

#Create the networkgraph object
def create_graph(nodeData, edgeData):

    ## Initiate the graph object
    G = nx.Graph()
    
    ## Tranform the data into the correct format for use with NetworkX
    #Node tuples (ID, dict of attributes)
    idList = nodeData['id'].tolist()
    labels =  pd.DataFrame(nodeData['Label'])
    labelDicts = labels.to_dict(orient='records')
    nodeTuples = [tuple(r) for r in zip(idList,labelDicts)]
    
    # Edge tuples (Source, Target, dict of attributes)
    sourceList = edgeData['Source'].tolist()
    targetList = edgeData['Target'].tolist()
    weights = pd.DataFrame(edgeData['Weight'])
    weightDicts = weights.to_dict(orient='records')
    edgeTuples = [tuple(r) for r in zip(sourceList,targetList,weightDicts)]
    
    ## Add the nodes and edges to the graph
    G.add_nodes_from(nodeTuples)
    G.add_edges_from(edgeTuples)
    G.remove_nodes_from(list(nx.isolates(G)))
    
    return G

def create_analysis(G):

    e_cent = nx.eigenvector_centrality(G)
    page_rank = nx.pagerank(G)
    degree = nx.degree(G)
    between = nx.betweenness_centrality(G)
        # Extract the analysis output and convert to a suitable scale and format
    e_cent_size = pd.DataFrame.from_dict(e_cent, orient='index',
                                         columns=['cent_value'])
    e_cent_size = e_cent_size['cent_value'] 
    e_cent_size.reset_index(drop=True, inplace=True)
    page_rank_size = pd.DataFrame.from_dict(page_rank, orient='index',
                                            columns=['rank_value'])
    page_rank_size = page_rank_size['rank_value'] 
    page_rank_size.reset_index(drop=True, inplace=True)
    
    degree_list = list(degree)
    degree_dict = dict(degree_list)
    degree_size = pd.DataFrame.from_dict(degree_dict, orient='index',
                                         columns=['deg_value'])
    degree_size = degree_size['deg_value'] 
    
    degree_size.reset_index(drop=True, inplace=True)
    between_size = pd.DataFrame.from_dict(between, orient='index',
                                          columns=['betw_value'])
    between_size = between_size['betw_value'] 
    between_size.reset_index(drop=True, inplace=True)

    dfs = [e_cent_size,page_rank_size,degree_size,between_size]
    df = pd.concat(dfs, axis=1)

    df = pd.concat(dfs, axis=1)
    cols = list(df.columns)
    an_arr = df.to_numpy(copy=True)

    scaler = MinMaxScaler()
    an_scaled = scaler.fit_transform(an_arr)
    an_df = pd.DataFrame(an_scaled)

    an_df.columns = cols
    an_mins = list(an_df.min())
    #Set the centrality rankings between 1 to 10. The reason for 1 is the makes more sense than 0 
    for i in range(len(an_mins)):
        an_df[cols[i]] *= 9
        an_df[cols[i]] -= an_mins[i] - 1
    ####
    an_df.columns = ['cent_st_val', 'rank_st_val', 'deg_st_val', 'betw_st_val']


    #### bringing together original and scaled versions of dataframes
    full_an_df = pd.concat([df, an_df], axis=1)
    
    return full_an_df

#create the Cytograph object for the dash layout. 
def cytograph( nodes, edges, G, analysis):
    nodes = nodes.astype(str)
    edges = edges.astype(str)
    nodes = nodes[nodes.id.isin(list(G))]
    
    nodes_list = list()
    for i in range(len(nodes)):
        nodes_data = {
            "data" : {"id":nodes.iloc[i,0],
                        "label":nodes.iloc[i,1],
                        "description":nodes.iloc[i,2],
                        "centrality":round(analysis.iloc[i,4],1),
                        "ranking":round(analysis.iloc[i,5],1),
                        "degree":round(analysis.iloc[i,6],1),
                        "betweenness":round(analysis.iloc[i,7],1) }    
                }
        nodes_list.append(nodes_data)
    edges_list = list()
    for j in range(len(edges)):
        edge_data = {
                "data": {"source": edges.iloc[j,0], 
                         "target": edges.iloc[j,1],
                         "weight": edges.iloc[j,2]
                        }
            }
        edges_list.append(edge_data)
    
    elements = nodes_list + edges_list
    return elements