import pandas as pd
import numpy as np
from prophet import Prophet

# 加载数据
prophet_results_path = 'prophet_results_all_products.csv'
data_path = 'Processed_Cleaned_and_Merged_Data.csv'

prophet_results = pd.read_csv(prophet_results_path)
data = pd.read_csv(data_path)

# 确保日期格式正确
prophet_results['transaction_date'] = pd.to_datetime(prophet_results['transaction_date'])
data['transaction_date'] = pd.to_datetime(data['transaction_date'])

# 选择需要外推的日期范围
target_dates = pd.date_range(start='2022-11-10', end='2022-11-23')

# 提取每个产品的数据
product_ids = data['product_pid'].unique()

# 准备结果存储
forecast_results = []

# 定义函数：对残差进行随机采样预测
def predict_residuals_from_distribution(residuals, size):
    if len(residuals) == 0:
        return np.zeros(size)  # 如果没有历史残差，用零填充
    return np.random.choice(residuals, size=size, replace=True)

# 定义函数：预测趋势
def forecast_trend(df, column_name):
    prophet_df = df.rename(columns={'transaction_date': 'ds', column_name: 'y'})[['ds', 'y']]
    model = Prophet()
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=len(target_dates))
    future = future[future['ds'].isin(target_dates)]  # 修复空数据框问题

    if future.empty:
        raise ValueError(f"No valid future dates for prediction in target_dates: {target_dates}")

    forecast = model.predict(future)

    # 补全缺失日期
    forecast = forecast.set_index('ds').reindex(target_dates).reset_index()
    forecast = forecast.rename(columns={'index': 'transaction_date', 'yhat': column_name})

    return forecast[['transaction_date', column_name]]


# 对每个产品进行处理
for product_id in product_ids:
    if product_id == 36 or product_id == 127 or product_id == 154:
        print(f"Skipping product {product_id} due to error.")
        continue

    # 筛选出该产品的 Prophet 结果和原始数据
    product_results = prophet_results[prophet_results['product_pid'] == product_id]
    product_data = data[data['product_pid'] == product_id]

    # 提取历史趋势和残差
    apply_trend_history = product_results[['transaction_date', 'apply_amt_trend']]
    apply_residual_history = product_results['apply_amt_residual'].dropna()

    redeem_trend_history = product_results[['transaction_date', 'redeem_amt_trend']]
    redeem_residual_history = product_results['redeem_amt_residual'].dropna()

    # 使用 Prophet 模型对 apply 和 redeem 进行趋势外推
    try:
        apply_trend_forecast = forecast_trend(apply_trend_history, 'apply_amt_trend')
        redeem_trend_forecast = forecast_trend(redeem_trend_history, 'redeem_amt_trend')
    except ValueError as e:
        print(f"Skipping product {product_id} due to error: {e}")
        continue

    # 对残差进行随机采样预测
    apply_residual_forecast = predict_residuals_from_distribution(apply_residual_history, len(target_dates))
    redeem_residual_forecast = predict_residuals_from_distribution(redeem_residual_history, len(target_dates))
    print(f"Product ID: {product_id}")
    print(f"Target Dates Length: {len(target_dates)}")
    print(f"Apply Trend Forecast Length: {len(apply_trend_forecast['apply_amt_trend'])}")
    print(f"Apply Residual Forecast Length: {len(apply_residual_forecast)}")
    print(f"Redeem Trend Forecast Length: {len(redeem_trend_forecast['redeem_amt_trend'])}")
    print(f"Redeem Residual Forecast Length: {len(redeem_residual_forecast)}")

    # 构造结果 DataFrame
    product_forecast = pd.DataFrame({
        'product_pid': product_id,
        'transaction_date': target_dates,
        'apply_amt_trend': apply_trend_forecast['apply_amt_trend'],
        'apply_amt_residual': apply_residual_forecast,
        'redeem_amt_trend': redeem_trend_forecast['redeem_amt_trend'],
        'redeem_amt_residual': redeem_residual_forecast
    })

    # 添加到总结果
    forecast_results.append(product_forecast)

# 合并所有结果
final_forecast = pd.concat(forecast_results, ignore_index=True)

# 保存结果
final_forecast.to_csv('prophet_forecast_20221110_20221123.csv', index=False)
print("预测完成，结果已保存为 'prophet_forecast_20221110_20221123.csv'")
