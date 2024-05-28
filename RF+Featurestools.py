import pandas as pd
import featuretools as ft
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # 加载数据
    logging.info("加载训练数据。")
    train_data = pd.read_csv('resource/train.csv', index_col='序号')

    # 填充缺失值
    logging.info("填充缺失值。")
    train_data = train_data.apply(lambda col: col.fillna('None') if col.dtype == 'object' else col.fillna(0))

    # 创建Featuretools实体集
    logging.info("创建Featuretools实体集。")
    es = ft.EntitySet(id='Missions')
    es.add_dataframe(dataframe_name='data', dataframe=train_data, index='序号', make_index=False)

    # 运行DFS生成新特征
    logging.info("运行深度特征合成。")
    feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name='data',
                                          trans_primitives=[], max_depth=1,
                                          ignore_columns={'data': ['突变名称']})

    # 预处理类别特征
    logging.info("设置预处理流水线。")
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'),
             list(feature_matrix.select_dtypes(include=['object', 'category']).columns))
        ],
        remainder='passthrough'
    )

    # 划分数据
    X = feature_matrix
    y = train_data['评级']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 定义参数网格
    param_grid = {
        'classifier__n_estimators': [200, 400, 600],
        'classifier__max_depth': [20, 40, 60],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }

    # 创建和训练模型
    logging.info("设置和训练模型流水线。")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 使用GridSearchCV进行超参数调优
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # 获取最佳模型
    best_model = grid_search.best_estimator_

    # 保存最佳模型
    joblib.dump(best_model, 'model/best_trained_model.pkl')

    # 验证模型
    val_predictions = best_model.predict(X_val)
    logging.info(f"验证集准确率: {accuracy_score(y_val, val_predictions)}")

    # 测试模型
    test_predictions = best_model.predict(X_test)
    logging.info(f"测试集准确率: {accuracy_score(y_test, test_predictions)}")

except Exception as e:
    logging.error(f"发生错误: {e}")
    raise


# 预测函数
def load_and_predict(new_data_path, output_csv_path):
    try:
        # 加载最佳模型
        logging.info("加载训练好的模型。")
        model = joblib.load('model/best_trained_model.pkl')

        # 加载新数据
        logging.info("加载新数据。")
        new_data = pd.read_csv(new_data_path, index_col='序号')
        new_data = new_data.apply(lambda col: col.fillna('None') if col.dtype == 'object' else col.fillna(0))

        # 确保数据类型匹配
        for col in new_data.columns:
            if col in train_data.columns:
                new_data[col] = new_data[col].astype(train_data[col].dtype)

        # 创建Featuretools实体集
        logging.info("为新数据创建Featuretools实体集。")
        es = ft.EntitySet(id='Missions')
        es.add_dataframe(dataframe_name='data', dataframe=new_data, index='序号', make_index=False)

        # 生成特征
        logging.info("为新数据运行深度特征合成。")
        feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name='data',
                                              trans_primitives=[], max_depth=1,
                                              ignore_columns={'data': ['突变名称']})

        # 确保特征矩阵的列与训练数据一致
        feature_matrix = feature_matrix.reindex(columns=X.columns, fill_value=0)

        # 预测
        predictions = model.predict(feature_matrix)

        # 保存结果
        result_df = new_data.copy()
        result_df['评级'] = predictions
        result_df.to_csv(output_csv_path, encoding='utf-8-sig', index=False)

        logging.info(f"预测结果已保存至: {output_csv_path}")
        return result_df

    except Exception as e:
        logging.error(f"预测时发生错误: {e}")
        raise


# 示例用法
new_data_path = 'resource/self.csv'
output_csv_path = 'resource/self_predictions1.csv'
predictions_df = load_and_predict(new_data_path, output_csv_path)
print("预测结果已保存至:", output_csv_path)
