import pandas as pd
import featuretools as ft

def load_and_preprocess_data(file_path, index_col='序号'):
    data = pd.read_csv(file_path, index_col=index_col)
    data = data.apply(lambda col: col.fillna('None') if col.dtype == 'object' else col.fillna(0))
    return data

def create_feature_matrix(data):
    es = ft.EntitySet(id='Missions')
    es.add_dataframe(dataframe_name='data', dataframe=data, index='序号', make_index=False)
    feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name='data',
                                          trans_primitives=[], max_depth=1,
                                          ignore_columns={'data': ['突变名称']})
    return feature_matrix
