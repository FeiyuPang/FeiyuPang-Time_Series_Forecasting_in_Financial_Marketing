import pandas as pd

# Load Prophet results and the full dataset
prophet_results_path = 'prophet_results_all_products.csv'
processed_data_path = 'Processed_Cleaned_and_Merged_Data.csv'
prophet_results = pd.read_csv(prophet_results_path)
processed_data = pd.read_csv(processed_data_path)

# Ensure transaction_date is in datetime format
prophet_results['transaction_date'] = pd.to_datetime(prophet_results['transaction_date'])
processed_data['transaction_date'] = pd.to_datetime(processed_data['transaction_date'])

# Merge Prophet results with the full dataset
merged_data = pd.merge(
    processed_data,
    prophet_results,
    on=['product_pid', 'transaction_date'],
    how='left'
)

# Create time-related features
merged_data['day_of_week'] = merged_data['transaction_date'].dt.dayofweek
merged_data['is_month_end'] = merged_data['transaction_date'].dt.is_month_end
merged_data['quarter'] = merged_data['transaction_date'].dt.quarter

# Select and rename columns for the final feature set
feature_columns = [
    'product_pid', 'transaction_date',
    'apply_amt','redeem_amt',
    'day_of_week', 'is_month_end', 'quarter',
    'uv_fundown', 'uv_stableown', 'uv_fundopt', 'uv_fundmarket', 'uv_termmarket',
    'apply_amt_trend', 'apply_amt_residual', 'redeem_amt_trend', 'redeem_amt_residual',
    'yield'
]

# Extract relevant features
feature_matrix = merged_data[feature_columns]

# Save the feature matrix
feature_matrix.to_csv('feature_matrix222.csv', index=False)

print("Feature engineering complete. Feature matrix saved as 'feature_matrix.csv'.")
