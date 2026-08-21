import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_PATH = r"C:\Users\user\Downloads\Sustainflow Ai PROJECT"
RESULTS_PATH = os.path.join(PROJECT_PATH, "Results")
DATASETS_PATH = os.path.join(PROJECT_PATH, "Datasets")
MODEL_PATH = os.path.join(RESULTS_PATH, "fao_model.joblib")

print("="*60)
print("MODULE 1: FAO DATA - ERROR ANALYSIS")
print("="*60)

# Load model
print(f"\nLoading model from: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    exit(1)

model = joblib.load(MODEL_PATH)
print("Model loaded successfully")

# Load data
data_path = os.path.join(DATASETS_PATH, "Fao_merged_clean.csv")
print(f"\nLoading data from: {data_path}")

if not os.path.exists(data_path):
    print(f"ERROR: Data file not found at {data_path}")
    exit(1)

data = pd.read_csv(data_path)
print(f"Loaded {len(data):,} rows")

# Identify target and features
# Assuming the target column is 'target' or 'value' or similar
target_cols = ['target', 'value', 'y', 'label']
target = None
for col in target_cols:
    if col in data.columns:
        target = col
        break

if target is None:
    print("\nAvailable columns in data:")
    print(data.columns.tolist())
    print("\nPlease specify the target column name")
    exit(1)

print(f"\nTarget column: {target}")
print(f"Features: {len(data.columns) - 1} columns")

# Prepare features (exclude target and identifier columns)
exclude_cols = [target, 'id', 'date', 'timestamp', 'year', 'month', 'day']
features = [col for col in data.columns if col not in exclude_cols]
print(f"Using {len(features)} features")

# For demonstration, use a subset or all data
# Split data into train/test if needed
np.random.seed(42)
split_idx = int(len(data) * 0.8)
train_data = data.iloc[:split_idx]
test_data = data.iloc[split_idx:]

print(f"\nTrain set: {len(train_data):,} rows")
print(f"Test set: {len(test_data):,} rows")

# Make predictions
print("\nMaking predictions...")
X_test = test_data[features]
y_test = test_data[target]

# Handle categorical features
for col in X_test.columns:
    if X_test[col].dtype == 'object':
        X_test[col] = X_test[col].astype('category')

predictions = model.predict(X_test)
print(f"Predictions made for {len(predictions):,} samples")

# Calculate metrics
print("\n" + "="*60)
print("MODEL PERFORMANCE METRICS")
print("="*60)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R^2:  {r2:.4f}")

# Calculate percentage error
mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-5))) * 100
print(f"MAPE: {mape:.2f}%")

# Create error dataframe
results_df = test_data.copy()
results_df['predicted'] = predictions
results_df['error'] = results_df[target] - predictions
results_df['abs_error'] = np.abs(results_df['error'])
results_df['pct_error'] = (results_df['abs_error'] / (results_df[target] + 1e-5)) * 100

# Error statistics
print("\n" + "="*60)
print("ERROR STATISTICS")
print("="*60)
print(f"Mean Absolute Error: {results_df['abs_error'].mean():.4f}")
print(f"Error Std Deviation: {results_df['error'].std():.4f}")
print(f"Max Error: {results_df['abs_error'].max():.4f}")
print(f"Min Error: {results_df['abs_error'].min():.4f}")
print(f"Median Absolute Error: {results_df['abs_error'].median():.4f}")

# Error percentiles
print("\nError Percentiles:")
print("-"*40)
for p in [25, 50, 75, 90, 95, 99]:
    print(f"{p}th percentile: {results_df['abs_error'].quantile(p/100):.4f}")

# Worst predictions
worst = results_df.nlargest(10, 'abs_error')
print("\n" + "="*60)
print("10 WORST PREDICTIONS")
print("="*60)
cols_to_show = [target, 'predicted', 'abs_error', 'pct_error']
available_cols = [col for col in cols_to_show if col in worst.columns]
print(worst[available_cols].to_string(index=False))

# Best predictions
best = results_df.nsmallest(10, 'abs_error')
print("\n" + "="*60)
print("10 BEST PREDICTIONS")
print("="*60)
print(best[available_cols].to_string(index=False))

# Create visualization plots
print("\nCreating visualization plots...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Actual vs Predicted
axes[0, 0].scatter(results_df[target], results_df['predicted'], alpha=0.2, s=1, color='blue')
max_val = max(results_df[target].max(), results_df['predicted'].max())
axes[0, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=2)
axes[0, 0].set_title('Actual vs Predicted', fontsize=12)
axes[0, 0].set_xlabel('Actual')
axes[0, 0].set_ylabel('Predicted')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Error Distribution
axes[0, 1].hist(results_df['error'], bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_title('Error Distribution', fontsize=12)
axes[0, 1].set_xlabel('Error')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Absolute Error Distribution
axes[0, 2].hist(results_df['abs_error'], bins=50, alpha=0.7, color='green', edgecolor='black')
axes[0, 2].set_title('Absolute Error Distribution', fontsize=12)
axes[0, 2].set_xlabel('Absolute Error')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Error vs Actual
axes[1, 0].scatter(results_df[target], results_df['error'], alpha=0.2, s=1, color='purple')
axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_title('Error vs Actual', fontsize=12)
axes[1, 0].set_xlabel('Actual')
axes[1, 0].set_ylabel('Error')
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Percentage Error Distribution
axes[1, 1].hist(results_df['pct_error'], bins=50, alpha=0.7, color='orange', edgecolor='black')
axes[1, 1].set_title('Percentage Error Distribution', fontsize=12)
axes[1, 1].set_xlabel('MAPE (%)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Residuals Q-Q plot
from scipy import stats
stats.probplot(results_df['error'], dist="norm", plot=axes[1, 2])
axes[1, 2].set_title('Q-Q Plot (Normality Check)', fontsize=12)
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle('FAO Model - Error Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plot_path = os.path.join(RESULTS_PATH, 'error_analysis_plots.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plots saved to: {plot_path}")
plt.show()

# Save predictions and analysis
predictions_path = os.path.join(RESULTS_PATH, 'fao_predictions.csv')
results_df.to_csv(predictions_path, index=False)
print(f"\nPredictions saved to: {predictions_path}")

# Save metrics to file
metrics_path = os.path.join(RESULTS_PATH, 'error_analysis_metrics.txt')
with open(metrics_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write("FAO MODEL - ERROR ANALYSIS REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Data: {data_path}\n")
    f.write(f"Total samples: {len(data):,}\n")
    f.write(f"Test samples: {len(test_data):,}\n\n")
    f.write("PERFORMANCE METRICS:\n")
    f.write("-"*40 + "\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"MAE:  {mae:.4f}\n")
    f.write(f"R^2:  {r2:.4f}\n")
    f.write(f"MAPE: {mape:.2f}%\n\n")
    f.write("ERROR STATISTICS:\n")
    f.write("-"*40 + "\n")
    f.write(f"Mean Absolute Error: {results_df['abs_error'].mean():.4f}\n")
    f.write(f"Error Std Deviation: {results_df['error'].std():.4f}\n")
    f.write(f"Max Error: {results_df['abs_error'].max():.4f}\n")
    f.write(f"Min Error: {results_df['abs_error'].min():.4f}\n")
    f.write(f"Median Absolute Error: {results_df['abs_error'].median():.4f}\n\n")
    f.write("ERROR PERCENTILES:\n")
    f.write("-"*40 + "\n")
    for p in [25, 50, 75, 90, 95, 99]:
        f.write(f"{p}th percentile: {results_df['abs_error'].quantile(p/100):.4f}\n")

print(f"Metrics saved to: {metrics_path}")

print("\n" + "="*60)
print("ERROR ANALYSIS COMPLETED")
print("="*60)
print(f"Model: {MODEL_PATH}")
print(f"Results: {RESULTS_PATH}")
print(f"Predictions: {predictions_path}")
print(f"Metrics: {metrics_path}")
print(f"Plots: {plot_path}")