import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib
import logging
from data_processing import load_and_preprocess_data, create_feature_matrix

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # 加载和处理数据
    logging.info("加载和处理训练数据。")
    train_data = load_and_preprocess_data('resource/train.csv')

    # 创建特征矩阵
    logging.info("创建特征矩阵。")
    feature_matrix = create_feature_matrix(train_data)

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
