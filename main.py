import logging
from predict_model import load_and_predict

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 示例用法
new_data_path = 'resource/self.csv'
output_csv_path = 'resource/self_predictions1.csv'
predictions_df = load_and_predict(new_data_path, output_csv_path)
print("预测结果已保存至:", output_csv_path)
