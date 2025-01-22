import pandas as pd
import numpy as np
from prophet import Prophet

# 加载数据
# prophet_results_path = 'prophet_results_all_products.csv'
data_path = 'Processed_Cleaned_and_Merged_Data.csv'
data = pd.read_csv(data_path)

# 确保日期格式正确
data['transaction_date'] = pd.to_datetime(data['transaction_date'])

# 选择需要外推的日期范围
target_dates = pd.date_range(start='2022-11-10', end='2022-11-23')

# 提取每个产品的数据
product_ids = data['product_pid'].unique()

# 准备结果存储
forecast_results = []

# 定义函数：预测趋势
def forecast_UV(df, column_name):
    # 准备 Prophet 数据
    prophet_df = df.rename(columns={'transaction_date': 'ds', column_name: 'y'})[['ds', 'y']]
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(prophet_df)

    # 生成未来数据框
    future = model.make_future_dataframe(periods=len(target_dates))
    future = future[future['ds'].isin(target_dates)]  # 确保未来日期与目标一致

    if future.empty:
        raise ValueError(f"No valid future dates for prediction in target_dates: {target_dates}")

    # 预测结果
    forecast = model.predict(future)

    # 检查预测结果是否包含必要的列
    if 'yhat' not in forecast.columns:
        raise KeyError("The forecast DataFrame does not contain 'yhat' column. Check model.predict output.")

    # 重命名列
    forecast = forecast.set_index('ds').reindex(target_dates).reset_index()
    forecast = forecast.rename(columns={'index': 'transaction_date', 'yhat': column_name})

    return forecast[['transaction_date', column_name]]

# 对每个产品进行处理
for product_id in product_ids:
    if product_id == 36 or product_id == 127 or product_id == 154 or product_id == 74 or product_id == 139:
        print(f"Skipping product {product_id} due to error.")
        continue

    # 筛选出该产品的 Prophet 结果和原始数据
    product_data = data[data['product_pid'] == product_id]

    # # 提取历史趋势和残差
    UVfundown_history = product_data[['transaction_date', 'uv_fundown']]
    UVstableown_history = product_data[['transaction_date', 'uv_stableown']]
    UVfundopt_history = product_data[['transaction_date', 'uv_fundopt']]
    UVfundmarket_history = product_data[['transaction_date', 'uv_fundmarket']]
    UVtermmarket_history = product_data[['transaction_date', 'uv_termmarket']]

    # 使用 Prophet 模型对 apply 和 redeem 进行趋势外推
    try:
        uv_fundown_forecast = forecast_UV(UVfundown_history, 'uv_fundown')
        uv_stableown_forecast = forecast_UV(UVstableown_history, 'uv_stableown')
        uv_fundopt_forecast = forecast_UV(UVfundopt_history, 'uv_fundopt')
        uv_fundmarket_forecast = forecast_UV(UVfundmarket_history, 'uv_fundmarket')
        uv_termmarket_forecast = forecast_UV(UVtermmarket_history, 'uv_termmarket')
    except ValueError as e:
        print(f"Skipping product {product_id} due to error: {e}")
        continue

    # 构造结果 DataFrame
    product_forecast = pd.DataFrame({
        'product_pid': product_id,
        'transaction_date': target_dates,
        'uv_fundown': uv_fundown_forecast['uv_fundown'],
        'uv_stableown': uv_stableown_forecast['uv_stableown'],
        'uv_fundopt': uv_fundopt_forecast['uv_fundopt'],
        'uv_fundmarket': uv_fundmarket_forecast['uv_fundmarket'],
        'uv_termmarket': uv_termmarket_forecast['uv_termmarket']
    })

    # 添加到总结果
    forecast_results.append(product_forecast)

# 合并所有结果
final_forecast = pd.concat(forecast_results, ignore_index=True)

# 保存结果
final_forecast.to_csv('UV_prophet_forecast_20221110_20221123.csv', index=False)
print("预测完成，结果已保存为 'UV_prophet_forecast_20221110_20221123.csv'")
