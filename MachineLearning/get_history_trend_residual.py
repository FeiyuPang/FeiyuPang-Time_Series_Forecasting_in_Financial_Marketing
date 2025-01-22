import pandas as pd
from prophet import Prophet

# Load the processed data
file_path = 'Processed_Cleaned_and_Merged_Data.csv'
data = pd.read_csv(file_path)

# Ensure transaction_date is in datetime format
data['transaction_date'] = pd.to_datetime(data['transaction_date'])

# Helper function to fit Prophet and extract components
def fit_prophet_and_extract(df):
    model = Prophet()
    model.fit(df)

    # Forecast
    future = model.make_future_dataframe(periods=0)
    forecast = model.predict(future)

    # Extract trend and residuals
    trend = forecast[['ds', 'trend']]

    # Ensure ds alignment before residual calculation
    forecast = forecast[forecast['ds'].isin(df['ds'])]
    df = df[df['ds'].isin(forecast['ds'])]

    # Align and compute residuals
    merged = pd.merge(df, forecast[['ds', 'yhat']], on='ds', how='inner')
    residuals = merged['y'] - merged['yhat']

    return trend, residuals

# Prepare an empty DataFrame for results
all_results = []

# Process each product individually
for product_id in data['product_pid'].unique():
    # Filter data for the current product
    product_data = data[data['product_pid'] == product_id]

    # Prepare data for apply_amt and redeem_amt
    apply_amt_df = product_data[['transaction_date', 'apply_amt']].copy()
    apply_amt_df.columns = ['ds', 'y']
    redeem_amt_df = product_data[['transaction_date', 'redeem_amt']].copy()
    redeem_amt_df.columns = ['ds', 'y']

    # Fit Prophet for apply_amt
    apply_trend, apply_residuals = fit_prophet_and_extract(apply_amt_df)

    # Fit Prophet for redeem_amt
    redeem_trend, redeem_residuals = fit_prophet_and_extract(redeem_amt_df)

    # Merge results into a single DataFrame
    result = pd.DataFrame({
        'product_pid': product_id,
        'transaction_date': apply_trend['ds'],
        'apply_amt_trend': apply_trend['trend'],
        'apply_amt_residual': apply_residuals,
        'redeem_amt_trend': redeem_trend['trend'],
        'redeem_amt_residual': redeem_residuals
    })

    # Append the result for this product
    all_results.append(result)

# Concatenate all results into a single DataFrame
final_results = pd.concat(all_results, ignore_index=True)

# Save the results
final_results.to_csv('prophet_results_all_products.csv', index=False)

print("Prophet modeling and extraction complete for all products. Results saved.")
