import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Load the feature matrix
data_path = 'feature_matrix222.csv'
data = pd.read_csv(data_path)

# data = data[data['product_pid'] == 2]

# Prepare features and target variables
features = data[['product_pid','day_of_week', 'is_month_end', 'quarter',
                 'uv_fundown', 'uv_fundopt',
                 'apply_amt_trend', 'apply_amt_residual', 'redeem_amt_trend', 'redeem_amt_residual',
                 'yield']]
apply_target = data['apply_amt']
redeem_target = data['redeem_amt']

# Split data into training and testing sets
X_train, X_test, y_apply_train, y_apply_test = train_test_split(features, apply_target, test_size=0.2, random_state=42)
_, _, y_redeem_train, y_redeem_test = train_test_split(features, redeem_target, test_size=0.2, random_state=42)

# Train Random Forest models for apply_amt and redeem_amt
apply_model = RandomForestRegressor(random_state=42)
apply_model.fit(X_train, y_apply_train)

redeem_model = RandomForestRegressor(random_state=42)
redeem_model.fit(X_train, y_redeem_train)

# Make predictions on the test set
apply_predictions = apply_model.predict(X_test)
redeem_predictions = redeem_model.predict(X_test)

# Evaluate model performance
apply_mae = mean_absolute_error(y_apply_test, apply_predictions)
apply_rmse = np.sqrt(mean_squared_error(y_apply_test, apply_predictions))

redeem_mae = mean_absolute_error(y_redeem_test, redeem_predictions)
redeem_rmse = np.sqrt(mean_squared_error(y_redeem_test, redeem_predictions))

# Compute net_in_amt predictions
net_in_predictions = apply_predictions - redeem_predictions
net_in_actual = y_apply_test.values - y_redeem_test.values

wmape_apply = np.sum(np.abs(apply_predictions - y_apply_test)) / np.sum(np.abs(y_apply_test))
wmape_redeem = np.sum(np.abs(redeem_predictions - y_redeem_test)) / np.sum(np.abs(y_redeem_test))

# Print evaluation metrics
print("Apply Amount:")
print(f"MAE: {apply_mae:.4f}, RMSE: {apply_rmse:.4f}, WMAPE: {wmape_apply:.4f}")
print("Redeem Amount:")
print(f"MAE: {redeem_mae:.4f}, RMSE: {redeem_rmse:.4f}, WMAPE: {wmape_redeem:.4f}")

# Save predictions
predictions_df = pd.DataFrame({
    'product_pid': data.loc[X_test.index, 'product_pid'],
    'transaction_date': data.loc[X_test.index, 'transaction_date'],
    'apply_amt_pred': apply_predictions,
    'redeem_amt_pred': redeem_predictions,
    'net_in_amt_pred': net_in_predictions
})

# 写文件
# predictions_df.to_csv('final_predictions.csv', index=False)
# print("Predictions saved as 'final_predictions222.csv'.")

# Visualization
product_id = int(input("Enter the product ID to visualize (e.g., 1): "))
product_data = predictions_df[predictions_df['product_pid'] == product_id]

if not product_data.empty:
    # Apply Amount Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(product_data['transaction_date'], y_apply_test.loc[product_data.index], label="Actual Apply Amount", marker='o')
    plt.plot(product_data['transaction_date'], product_data['apply_amt_pred'], label="Predicted Apply Amount", marker='x')
    plt.title(f"Apply Amount for Product {product_id}")
    plt.xlabel("Transaction Date")
    plt.ylabel("Apply Amount")
    plt.legend()
    plt.grid()
    plt.show()

    # Redeem Amount Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(product_data['transaction_date'], y_redeem_test.loc[product_data.index], label="Actual Redeem Amount", marker='o')
    plt.plot(product_data['transaction_date'], product_data['redeem_amt_pred'], label="Predicted Redeem Amount", marker='x')
    plt.title(f"Redeem Amount for Product {product_id}")
    plt.xlabel("Transaction Date")
    plt.ylabel("Redeem Amount")
    plt.legend()
    plt.grid()
    plt.show()
else:
    print(f"No data available for product ID {product_id}.")


final_data_path = 'final_features_matrix.csv'
final_data = pd.read_csv(final_data_path)


# Prepare features and target variables
features_to_predict = final_data[['product_pid','day_of_week', 'is_month_end', 'quarter',
                                  'uv_fundown', 'uv_fundopt',
                                  'apply_amt_trend', 'apply_amt_residual', 'redeem_amt_trend', 'redeem_amt_residual',
                                  'yield']]

# apply_target_to_predict = final_data['apply_amt']
# redeem_target = final_data['redeem_amt']

final_apply_amt = apply_model.predict(features_to_predict)
final_redeem_amt = redeem_model.predict(features_to_predict)
final_net_in_amt = final_apply_amt - final_redeem_amt

predictions_df = pd.DataFrame({
    'product_pid': final_data.loc[features_to_predict.index, 'product_pid'].apply(lambda x: f"product{x}"),
    'transaction_date': final_data.loc[features_to_predict.index, 'transaction_date'],
    'apply_amt_pred': final_apply_amt,
    'redeem_amt_pred': final_redeem_amt,
    'net_in_amt_pred': final_net_in_amt
})

predictions_df['transaction_date'] = pd.to_datetime(predictions_df['transaction_date'])

valid_date_ranges = [
    ('2022-11-10', '2022-11-11'),
    ('2022-11-14', '2022-11-18'),
    ('2022-11-21', '2022-11-23')
]

filtered_predictions = pd.concat([
    predictions_df[
        (predictions_df['transaction_date'] >= start_date) &
        (predictions_df['transaction_date'] <= end_date) &
        (predictions_df['transaction_date'].dt.weekday < 5)  # Weekday < 5 means Monday to Friday
    ]
    for start_date, end_date in valid_date_ranges
])

filtered_predictions['transaction_date'] = filtered_predictions['transaction_date'].dt.strftime('%Y%m%d')

filtered_predictions.to_csv('final_predictions.csv', index=False)
print("Predictions saved as 'final_predictions.csv'.")