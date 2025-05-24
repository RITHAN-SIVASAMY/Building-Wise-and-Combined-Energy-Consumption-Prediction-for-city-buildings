import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import time
import shap
import os

# Ensure Matplotlib uses an interactive backend
plt.ion()  # Enable interactive mode for display

# Start timing
start_time = time.time()

# Load data
df = pd.read_csv(r"C:\Users\ritha\OneDrive\Desktop\2ND SEMESTER\EEE\UK\Merged_Dataset.csv")
df["day"] = pd.to_datetime(df["day"], dayfirst=True)
df = df.set_index('day')
print("Total buildings:", len(df["building_id"].unique()))

# Check for NaNs in raw data
print("NaNs in raw data:", df.isna().sum().sum())

# Fill NaNs globally
df = df.fillna(method='ffill').fillna(method='bfill')
if df.isna().sum().sum() > 0:
    raise ValueError("NaNs persist in dataset after global fill")

# Select sample buildings
unique_buildings = df["building_id"].unique()
sample_buildings = unique_buildings[:1637]  # Test with 10; change to [:1637] for all
print(f"Processing {len(sample_buildings)} buildings: {sample_buildings}")

# Features
feature_cols = ['energy_mean', 'energy_median', 'energy_max', 'energy_min', 'energy_std',
                'temperatureMax', 'temperatureMin', 'humidity', 'windSpeed', 'pressure', 'cloudCover']

r2_scores = []
rmse_scores = []
feature_importances = []
shap_values_list = []
skipped_buildings = []  # Track buildings with unusual R²
all_y_test = []  # Collect actual values for aggregated plots
all_y_pred = []  # Collect predicted values for aggregated plots
all_dates = []  # Collect dates for line plot

R2_THRESHOLD = 0.7  # Neglect buildings with R² < 0.7

# Create output directory for plots
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

for building_id in sample_buildings:
    print(f"\nProcessing Building {building_id}...")
    building_data = df[df["building_id"] == building_id].copy()

    # Prepare features and target
    features = building_data[feature_cols]
    target = building_data['energy_sum']

    # Check for NaNs in target
    if target.isna().sum() > 0:
        print(f"Skipping Building {building_id} - NaNs in target ({target.isna().sum()})")
        continue

    # Add time-based features
    features['day_of_week'] = building_data.index.dayofweek
    features['month'] = building_data.index.month
    features['day_of_year'] = building_data.index.dayofyear

    # Fill missing values in features
    features = features.fillna(method='ffill').fillna(method='bfill')
    if features.isna().sum().sum() > 0:
        print(f"Skipping Building {building_id} - NaNs in features after fill ({features.isna().sum().sum()})")
        continue

    # Skip if insufficient data
    if len(features) < 20:
        print(f"Skipping Building {building_id} - insufficient data ({len(features)} rows)")
        continue

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, shuffle=False)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Check for NaNs after scaling
    if np.any(np.isnan(X_train_scaled)) or np.any(np.isnan(X_test_scaled)):
        print(f"Skipping Building {building_id} - NaNs after scaling")
        continue

    # Train XGBoost model
    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.25, max_depth=5, subsample=0.8, 
                             colsample_bytree=0.8, random_state=37, reg_lambda=1.0)
    model.fit(X_train_scaled, y_train)

    # Predict and compute R² and RMSE
    y_pred = model.predict(X_test_scaled)
    if np.any(np.isnan(y_pred)):
        print(f"Skipping Building {building_id} - NaNs in predictions")
        continue

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Baseline R²
    y_mean = np.full_like(y_test, y_train.mean())
    r2_baseline = r2_score(y_test, y_mean)
    
    # Filter unusual R²
    if r2 < R2_THRESHOLD:
        print(f"Building {building_id} - Unusual R² detected. R²: {r2:.3f}, Baseline R²: {r2_baseline:.3f}, RMSE: {rmse:.3f}")
        skipped_buildings.append((building_id, r2, rmse))
    else:
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        feature_importances.append(model.feature_importances_)
        # Ensure valid data before appending
        if not y_test.isna().any() and not np.isnan(y_pred).any():
            all_y_test.extend(y_test.values)
            all_y_pred.extend(y_pred)
            all_dates.extend(X_test.index)
        else:
            print(f"Warning: Skipped adding data for Building {building_id} to aggregated plots due to NaNs")
        
        # --- SHAP Analysis ---
        try:
            explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
            shap_values = explainer.shap_values(X_test_scaled)
            shap_values_list.append((building_id, X_test_scaled, shap_values, features.columns))
        except Exception as e:
            print(f"SHAP failed for Building {building_id}: {str(e)}")
    
    print(f"Building {building_id} - R²: {r2:.3f}, Baseline R²: {r2_baseline:.3f}, RMSE: {rmse:.3f}")

# --- Aggregated Actual vs Predicted Line Plot ---
if len(all_y_test) > 0 and len(all_y_pred) > 0 and len(all_dates) > 0:
    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'Date': all_dates,
        'Actual': all_y_test,
        'Predicted': all_y_pred
    })
    # Sort by date for continuous line
    filtered_plot_data = plot_data.sort_values('Date')
    print(f"Generating aggregated line plot with {len(filtered_plot_data)} points from {len(r2_scores)} buildings")
    
    if len(filtered_plot_data) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(filtered_plot_data['Date'], filtered_plot_data['Actual'], label='Actual Energy Sum', color='blue', alpha=0.7)
        plt.plot(filtered_plot_data['Date'], filtered_plot_data['Predicted'], label='Predicted Energy Sum', color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Date')
        plt.ylabel('Energy Sum (kWh)')
        plt.title(f'Actual vs Predicted Energy Sum Across All Buildings (XGBoost)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'actual_vs_predicted_all_buildings_line_energy_sum_xgb_F.png')
        plt.savefig(plot_path)
        print(f"Saved Actual vs Predicted line plot: {plot_path}")
        plt.show()
        plt.close()
    else:
        print("No points remain for line plot")

# --- Aggregated Actual vs Predicted Scatter Plot ---
if len(all_y_test) > 0 and len(all_y_pred) > 0:
    all_y_test = np.array(all_y_test)
    all_y_pred = np.array(all_y_pred)
    mask = (all_y_test >= 0) & (all_y_test <= 0.5)
    filtered_y_test = all_y_test[mask]
    filtered_y_pred = all_y_pred[mask]
    print(f"Generating aggregated scatter plot with {len(filtered_y_test)} points (after filtering {len(all_y_test) - len(filtered_y_test)} points outside 0 to 0.5 kWh) from {len(r2_scores)} buildings")
    
    if len(filtered_y_test) > 0:
        plt.figure(figsize=(8, 6))
        plt.scatter(filtered_y_test, filtered_y_pred, color='blue', alpha=0.5, label='Actual vs Predicted')
        plt.plot([filtered_y_test.min(), filtered_y_test.max()], [filtered_y_test.min(), filtered_y_test.max()], 
                 'r--', label='y=x (Perfect Prediction)')
        plt.xlabel('Actual Energy Sum (kWh)')
        plt.ylabel('Predicted Energy Sum (kWh)')
        plt.title(f'Actual vs Predicted Energy Sum (0 to 0.5 kWh, XGBoost)')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'actual_vs_predicted_all_buildings_energy_sum_0to0_5_xgb_F.png')
        plt.savefig(plot_path)
        print(f"Saved Actual vs Predicted scatter plot: {plot_path}")
        plt.show()
        plt.close()
    else:
        print("No points remain after applying 0 to 0.5 kWh filter for scatter plot")

# --- Aggregated Regression Plot ---
if len(all_y_test) > 0 and len(all_y_pred) > 0:
    if len(filtered_y_test) > 0:
        plt.figure(figsize=(8, 6))
        sns.regplot(x=filtered_y_test, y=filtered_y_pred, scatter_kws={'color': 'blue', 'alpha': 0.5}, 
                    line_kws={'color': 'red'}, label='Regression Line')
        plt.xlabel('Actual Energy Sum (kWh)')
        plt.ylabel('Predicted Energy Sum (kWh)')
        plt.title(f'Regression Plot for Energy Sum (0 to 0.5 kWh, XGBoost)')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'regression_plot_all_buildings_energy_sum_0to0_5_xgb_F.png')
        plt.savefig(plot_path)
        print(f"Saved Regression plot: {plot_path}")
        plt.show()
        plt.close()
    else:
        print("No points remain after applying 0 to 0.5 kWh filter for regression plot")
else:
    print("No data available for aggregated plots (all buildings may have R² < 0.7)")

# --- SHAP Visualization ---
if shap_values_list:
    # Summary Plot
    all_shap_values = np.vstack([sv[2] for sv in shap_values_list])
    all_X_test = np.vstack([sv[1] for sv in shap_values_list])
    feature_names = shap_values_list[0][3]
    plt.figure(figsize=(10, 6))
    shap.summary_plot(all_shap_values, all_X_test, feature_names=feature_names, show=False)
    plt.title("SHAP Summary Plot Across Buildings (XGBoost)")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'shap_summary_plot_energy_sum_xgb_F.png')
    plt.savefig(plot_path)
    print(f"Saved SHAP Summary plot: {plot_path}")
    plt.show()
    plt.close()

    # Force Plot (last valid building)
    last_building_id, last_X_test, last_shap_values, _ = shap_values_list[-1]
    shap.force_plot(explainer.expected_value, last_shap_values[0], last_X_test[0], 
                    feature_names=feature_names, matplotlib=True, show=False)
    plt.title(f"SHAP Force Plot for Building {last_building_id} (First Test Sample)")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'shap_force_plot_last_building_energy_sum_xgb_F.png')
    plt.savefig(plot_path)
    print(f"Saved SHAP Force plot: {plot_path}")
    plt.show()
    plt.close()

# --- Plotting Section ---
# R² Distribution
plt.figure(figsize=(8, 6))
plt.hist(r2_scores, bins=10, color='skyblue', edgecolor='black')
plt.xlabel('R² Score')
plt.ylabel('Number of Buildings')
plt.title('Distribution of R² Scores Across Buildings (XGBoost)')
plt.tight_layout()
plot_path = os.path.join(output_dir, 'r2_distribution_energy_sum_xgb_F.png')
plt.savefig(plot_path)
print(f"Saved R² Distribution plot: {plot_path}")
plt.show()
plt.close()

# RMSE Distribution
plt.figure(figsize=(8, 6))
plt.hist(rmse_scores, bins=10, color='lightgreen', edgecolor='black')
plt.xlabel('RMSE (kWh)')
plt.ylabel('Number of Buildings')
plt.title('Distribution of RMSE Across Buildings (XGBoost, R² ≥ 0.7)')
plt.tight_layout()
plot_path = os.path.join(output_dir, 'rmse_distribution_energy_sum_xgb_F.png')
plt.savefig(plot_path)
print(f"Saved RMSE Distribution plot: {plot_path}")
plt.show()
plt.close()

# Feature Importance
if feature_importances:
    mean_importance = np.mean(feature_importances, axis=0)
    feature_names = features.columns
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, mean_importance, color='lightcoral')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Average Feature Importance Across Buildings (XGBoost, R² ≥ 0.7)')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'feature_importance_energy_sum_xgb_F.png')
    plt.savefig(plot_path)
    print(f"Saved Feature Importance plot: {plot_path}")
    plt.show()
    plt.close()

# --- Aggregate R² and RMSE Analysis ---
print("\n=== Aggregate R² and RMSE Analysis (XGBoost, R² ≥ 0.7) ===")
if r2_scores and rmse_scores:
    mean_r2 = np.mean(r2_scores)
    min_r2 = np.min(r2_scores)
    max_r2 = np.max(r2_scores)
    median_r2 = np.median(r2_scores)
    mean_rmse = np.mean(rmse_scores)
    min_rmse = np.min(rmse_scores)
    max_rmse = np.max(rmse_scores)
    median_rmse = np.median(rmse_scores)
    print(f"Mean R²: {mean_r2:.3f}")
    print(f"Min R²: {min_r2:.3f}")
    print(f"Max R²: {max_r2:.3f}")
    print(f"Median R²: {median_r2:.3f}")
    print(f"Mean RMSE: {mean_rmse:.3f}")
    print(f"Min RMSE: {min_rmse:.3f}")
    print(f"Max RMSE: {max_rmse:.3f}")
    print(f"Median RMSE: {median_rmse:.3f}")
    print(f"Number of buildings analyzed: {len(r2_scores)}")
else:
    print("No valid R² or RMSE scores computed")

# Report skipped buildings
print("\n=== Skipped Buildings (R² < 0.7) ===")
if skipped_buildings:
    for b_id, b_r2, b_rmse in skipped_buildings:
        print(f"Building {b_id} - R²: {b_r2:.3f}, RMSE: {b_rmse:.3f}")
    print(f"Total skipped buildings: {len(skipped_buildings)}")
else:
    print("No buildings skipped due to unusual R²")

print(f"Total time taken: {(time.time() - start_time) / 60:.2f} minutes")