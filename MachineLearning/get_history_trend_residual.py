import pandas as pd
from prophet import Prophet

path = 'Processed_Cleaned_and_Merged_Data.csv'
data = pd.read_csv(path)

data['transaction_date'] = pd.to_datetime(data['transaction_date'])

def fit_prophet_and_extract(df):
    model = Prophet()
    model.fit(df)

    # 模型预测
    future = model.make_future_dataframe(periods=0)
    forecast = model.predict(future)

    # 获取趋势
    trend = forecast[['ds', 'trend']]   # 注意切片逗号后面必须有空格

    # 获取残差
    # 准备工作:日期对齐,由于prophet模型对节假日的处理,会导致日期不对齐,后续计算残差的时候出现错误
    forecast['ds'] = forecast[forecast['ds'].isin(df['ds'])]
    df['ds'] = df[df['ds'].isin(forecast['ds'])]

    # 不能直接使用df['y'] - forecast['yhat'], 可能日期顺序对不上
    merge = pd.merge(df, forecast['ds', 'yhat'], on='ds', how='inner')
    residual = merge['y'] - merge['yhat']

    return trend, residual

# 空列表存储结果
all_result = []

for product_id in data['product_pid'].unique():
    # 是一个dataframe 不是列表 这与使用几个中括号没有关系
    # 单中括号：如果中括号内是布尔条件或切片，返回 DataFrame。
    product_data = data[data['product_pid'] == product_id]

    # 准备训练数据(df)
    apply_amt_df = data[['transaction_date', 'apply_amt']].copy()
    # 改列名
    apply_amt_df.columns = ['ds', 'y']
    redeem_amt_df = data[['transaction_date', 'redeem_amt']].copy()
    redeem_amt_df.columns = ['ds', 'y']

    # 传入模型获得两个值的趋势残差
    apply_trend, apply_residuals = fit_prophet_and_extract(apply_amt_df)
    redeem_trend, redeem_residuals = fit_prophet_and_extract(redeem_amt_df)

    # 创建结果dataframe
    result_data = {
        'product_pid': product_id,
        'transaction_date': apply_trend['transaction_date'],
        'apply_amt_trend': apply_trend['trend'],
        'apply_amt_residual': apply_residuals,
        'redeem_amt_trend': redeem_trend['trend'],
        'redeem_residuals': redeem_residuals
    }
    result = pd.DataFrame(result_data)

    # 添加到列表中 列表中的元素是dataframe
    all_result.append(result)

final_result = pd.concat(all_result,ignore_index=True)

final_result.to_csv('prophet_results_all_products.csv', index=False)

print("Prophet modeling and extraction complete for all products. Result saved.")


