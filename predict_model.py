import pandas as pd
import joblib
import logging
from data_processing import load_and_preprocess_data, create_feature_matrix

def load_and_predict(new_data_path, output_csv_path):
    try:
        # 加载模型
        logging.info("加载训练好的模型。")
        model = joblib.load('model/best_trained_model.pkl')

        # 加载和处理新数据
        logging.info("加载新数据。")
        new_data = load_and_preprocess_data(new_data_path)

        # 创建特征矩阵
        logging.info("创建特征矩阵。")
        feature_matrix = create_feature_matrix(new_data)

        # 确保特征矩阵的列与训练数据一致
        train_data = load_and_preprocess_data('resource/train.csv')
        train_feature_matrix = create_feature_matrix(train_data)
        feature_matrix = feature_matrix.reindex(columns=train_feature_matrix.columns, fill_value=0)

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
if __name__ == "__main__":
    new_data_path = 'resource/self.csv'
    output_csv_path = 'resource/self_predictions1.csv'
    predictions_df = load_and_predict(new_data_path, output_csv_path)
    print("预测结果已保存至:", output_csv_path)
