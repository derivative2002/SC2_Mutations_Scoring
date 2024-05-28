import pandas as pd
import featuretools as ft
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

# 加载数据
train_data = pd.read_csv('resource/train.csv', index_col='序号')

# 填充缺失值，根据列的数据类型填充
train_data = train_data.apply(lambda col: col.fillna('None') if col.dtype == 'object' else col.fillna(0))

# 创建Featuretools实体集，不包含“突变名称”列作为特征
es = ft.EntitySet(id='Missions')
es.add_dataframe(dataframe_name='data', dataframe=train_data, index='序号', make_index=False)

# 运行深度特征合成（DFS）生成新特征
feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name='data',
                                      trans_primitives=[],
                                      max_depth=1,
                                      ignore_columns={'data': ['突变名称']})

# 使用OneHotEncoder处理类别特征
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'),
         list(feature_matrix.select_dtypes(include=['object', 'category']).columns))
    ],
    remainder='passthrough'
)

# 构建预处理和模型训练的流水线
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=400,
        max_depth=40,
        random_state=42,
        min_samples_split=5,
        min_samples_leaf=2
    ))
])

# 特征和标签
X = feature_matrix
y = train_data['评级']

# 划分训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 训练模型
pipeline.fit(X_train, y_train)

# 保存模型
joblib.dump(pipeline, 'model/trained_model.pkl')

# 验证模型
val_predictions = pipeline.predict(X_val)
print("Val accuracy:", accuracy_score(y_val, val_predictions))

# 测试模型
test_predictions = pipeline.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, test_predictions))


# 函数：加载模型并进行预测，并将结果保存到CSV
def load_and_predict(new_data_path, output_csv_path):
    # 加载模型
    model = joblib.load('model/trained_model.pkl')

    # 加载并处理新数据
    new_data = pd.read_csv(new_data_path, index_col='序号')
    new_data = new_data.apply(lambda col: col.fillna('None') if col.dtype == 'object' else col.fillna(0))

    # 确保数据类型匹配
    for col in new_data.columns:
        if col in train_data.columns:
            new_data[col] = new_data[col].astype(train_data[col].dtype)

    # 创建Featuretools实体集
    es = ft.EntitySet(id='Missions')
    es.add_dataframe(dataframe_name='data', dataframe=new_data, index='序号', make_index=False)

    # 调试信息：检查数据类型
    print("New data types:\n", new_data.dtypes)

    # 运行DFS生成新特征
    try:
        feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name='data',
                                              trans_primitives=[],
                                              max_depth=1,
                                              ignore_columns={'data': ['突变名称']})
    except Exception as e:
        print("DFS error:", e)
        raise

    # 确保特征矩阵的列与训练数据一致
    feature_matrix = feature_matrix.reindex(columns=X.columns, fill_value=0)

    # 预测
    predictions = model.predict(feature_matrix)

    # 创建带有预测结果的DataFrame
    result_df = new_data.copy()
    result_df['评级'] = predictions

    # 保存预测结果到CSV文件
    result_df.to_csv(output_csv_path, encoding='utf-8-sig', index=False)

    return result_df


# 示例：使用新数据进行预测并保存结果
new_data_path = 'resource/Alatao.csv'  # 替换为你的新数据路径
output_csv_path = 'resource/self_predictions.csv'  # 替换为你的输出路径
predictions_df = load_and_predict(new_data_path, output_csv_path)
print("Predictions saved to:", output_csv_path)
