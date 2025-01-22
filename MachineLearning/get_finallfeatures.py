import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


def merge_tables(uv_forecast_path, prophet_forecast_path, cbyieldcurve_path):
    """
    将三个表格按照 product_pid 和 transaction_date 合并，返回合并后的 DataFrame。

    参数:
    uv_forecast_path: str, UV Prophet Forecast 表格的路径
    prophet_forecast_path: str, Prophet Forecast 表格的路径
    cbyieldcurve_path: str, CB Yield Curve 表格的路径

    返回:
    DataFrame: 合并后的表格
    """
    # 加载表格
    uv_forecast = pd.read_csv(uv_forecast_path)
    prophet_forecast = pd.read_csv(prophet_forecast_path)
    cbyieldcurve = pd.read_csv(cbyieldcurve_path)

    # 对齐日期字段名
    cbyieldcurve = cbyieldcurve.rename(columns={'enddate': 'transaction_date'})

    # 确保日期字段格式一致
    uv_forecast['transaction_date'] = pd.to_datetime(uv_forecast['transaction_date'])
    prophet_forecast['transaction_date'] = pd.to_datetime(prophet_forecast['transaction_date'])
    cbyieldcurve['transaction_date'] = pd.to_datetime(cbyieldcurve['transaction_date'])

    # 按 product_pid 和 transaction_date 合并 UV 和 Prophet 表
    merged = pd.merge(uv_forecast, prophet_forecast, on=['product_pid', 'transaction_date'], how='inner')

    # 按 transaction_date 合并 Yield Curve 表
    merged = pd.merge(merged, cbyieldcurve, on='transaction_date', how='left')

    return merged


# 示例调用
uv_forecast_path = 'UV_prophet_forecast_20221110_20221123.csv'
prophet_forecast_path = 'prophet_forecast_20221110_20221123.csv'
cbyieldcurve_path = 'cbyieldcurve_info_final.csv'

merged_data = merge_tables(uv_forecast_path, prophet_forecast_path, cbyieldcurve_path)

feature_columns = [
    'product_pid', 'transaction_date',
    'uv_fundown', 'uv_stableown', 'uv_fundopt', 'uv_fundmarket', 'uv_termmarket',
    'apply_amt_trend', 'apply_amt_residual', 'redeem_amt_trend', 'redeem_amt_residual',
    'yield','day_of_week', 'is_month_end', 'quarter'
]

# Extract relevant features
feature_matrix = merged_data[feature_columns]

# Save the feature matrix
feature_matrix.to_csv('final_features_matrix.csv', index=False)

print("Feature engineering complete. Feature matrix saved as 'final_features_matrix.csv'.")