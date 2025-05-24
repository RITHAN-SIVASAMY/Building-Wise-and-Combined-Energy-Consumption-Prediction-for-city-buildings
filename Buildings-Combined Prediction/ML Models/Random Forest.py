import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import time
import shap
import os

# Ensure Matplotlib uses an interactive backend
plt.ion()  # Enable interactive mode for display

# Project Objective
print("🎯 Objective: Predict daily energy_sum across multiple buildings using Random Forest with SHAP analysis.")

# Load Dataset
file_path = r"C:\Users\ritha\OneDrive\Desktop\2ND SEMESTER\EEE\UK\Merged_Dataset.csv"
data = pd.read_csv(file_path)
data['day'] = pd.to_datetime(data['day'], dayfirst=True)
data = data.set_index('day')  # Set index for date access
print("✅ Dataset loaded! Shape:", data.shape)
print("Unique building_ids:", data['building_id'].nunique())

# Handle Missing Values Globally
data.ffill(inplace=True)
data.bfill(inplace=True)
print("✅ Missing values handled globally. NaNs remaining:", data.isnull().sum().sum())
if data.isnull().sum().sum() > 0:
    raise ValueError("NaNs persist in dataset after global fill")

# Encode Categorical Data (if any)
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
print("✅ Categorical variables encoded.")

# Create output directory for plots
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# --- Multiple Building Prediction with Random Forest and SHAP ---
print("\n=== Multiple Building Prediction (Random Forest with SHAP) ===")
valid_building_ids = data['building_id'].unique()
sample_building_ids = valid_building_ids[:1637]  # Test with 10; change to [:1637] for all
print(f"Processing {len(sample_building_ids)} buildings...")

r2_scores = []
rmse_scores = []
feature_importances = []
shap_values_list = []
skipped_buildings = []  # Track buildings with unusual R²
all_y_test = []  # Collect actual values for aggregated plots
all_y_pred = []  # Collect predicted values for aggregated plots
all_dates = []  # Collect dates for line plot
R2_THRESHOLD = 0.7  # Neglect buildings with R² < 0.7

model = RandomForestRegressor(n_estimators=100, random_state=42)

start_time = time.time()

for building_id in sample_building_ids:
    print(f"\nProcessing Building {building_id}...")
    building_data = data[data['building_id'] == building_id].copy()

    # Define Features and Target (numeric only)
    numeric_cols = building_data.select_dtypes(include=[np.number]).columns
    if 'energy_sum' not in building_data.columns:
        print(f"Skipping Building {building_id} - 'energy_sum' not found")
        continue
    X_ind = building_data[numeric_cols].drop(columns=['energy_sum'])
    y_ind = building_data['energy_sum']

    # Check for NaNs
    if y_ind.isna().sum() > 0:
        print(f"Skipping Building {building_id} - NaNs in target ({y_ind.isna().sum()})")
        continue
    X_ind = X_ind.ffill().bfill()
    if X_ind.isna().sum().sum() > 0:
        print(f"Skipping Building {building_id} - NaNs in features after fill ({X_ind.isna().sum().sum()})")
        continue

    # Skip if insufficient data
    if len(X_ind) < 10:
        print(f"Skipping Building {building_id} - insufficient data ({len(X_ind)} rows)")
        continue

    # Split Data
    X_train_ind, X_test_ind, y_train_ind, y_test_ind = train_test_split(
        X_ind, y_ind, test_size=0.2, random_state=42
    )

    # Store test indices for date access
    test_indices = X_test_ind.index

    # Feature Scaling
    scaler_ind = StandardScaler()
    X_train_ind = scaler_ind.fit_transform(X_train_ind)
    X_test_ind = scaler_ind.transform(X_test_ind)

    # Check for NaNs after scaling
    if np.any(np.isnan(X_train_ind)) or np.any(np.isnan(X_test_ind)):
        print(f"Skipping Building {building_id} - NaNs after scaling")
        continue

    # Train Random Forest
    model.fit(X_train_ind, y_train_ind)
    y_pred_ind = model.predict(X_test_ind)
    if np.any(np.isnan(y_pred_ind)):
        print(f"Skipping Building {building_id} - NaNs in predictions")
        continue

    # Calculate R² and RMSE
    r2 = r2_score(y_test_ind, y_pred_ind)
    rmse = np.sqrt(mean_squared_error(y_test_ind, y_pred_ind))
    
    # Filter unusual R²
    if r2 < R2_THRESHOLD:
        print(f"Building {building_id} - Unusual R² detected. R²: {r2:.4f}, RMSE: {rmse:.4f}")
        skipped_buildings.append((building_id, r2, rmse))
    else:
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        feature_importances.append(model.feature_importances_)
        # Ensure valid data before appending
        if not y_test_ind.isna().any() and not np.isnan(y_pred_ind).any():
            all_y_test.extend(y_test_ind.values)
            all_y_pred.extend(y_pred_ind)
            all_dates.extend(test_indices)
        else:
            print(f"Warning: Skipped adding data for Building {building_id} to aggregated plots due to NaNs")
        
        # --- SHAP Analysis ---
        try:
            explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
            shap_values = explainer.shap_values(X_test_ind)
            shap_values_list.append((building_id, X_test_ind, shap_values, X_ind.columns))
        except Exception as e:
            print(f"SHAP failed for Building {building_id}: {str(e)}")
    
    print(f"Building {building_id} - R²: {r2:.4f}, RMSE: {rmse:.4f}")

    # --- Future Prediction for Last Building ---
    if building_id == sample_building_ids[-1]:
        last_date = building_data.index.max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
        future_data = pd.DataFrame(index=future_dates, columns=X_ind.columns)
        future_data = future_data.fillna(X_ind.iloc[-1])
        future_data_scaled = scaler_ind.transform(future_data)
        future_predictions = model.predict(future_data_scaled)

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
        plt.title(f'Actual vs Predicted Energy Sum Across All Buildings (Random Forest)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'actual_vs_predicted_all_buildings_line_energy_sum_rf_F.png')
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
        plt.title(f'Actual vs Predicted Energy Sum (0 to 0.5 kWh, Random Forest)')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'actual_vs_predicted_all_buildings_energy_sum_0to0_5_rf_F.png')
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
        plt.title(f'Regression Plot for Energy Sum (0 to 0.5 kWh, Random Forest)')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'regression_plot_all_buildings_energy_sum_0to0_5_rf_F.png')
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
    plt.title("SHAP Summary Plot Across Buildings (Random Forest)")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'shap_summary_plot_energy_sum_rf_F.png')
    plt.savefig(plot_path)
    print(f"Saved SHAP Summary plot: {plot_path}")
    plt.show()
    plt.close()

    # Force Plot for Last Building
    last_building_id, last_X_test, last_shap_values, _ = shap_values_list[-1]
    shap.force_plot(explainer.expected_value, last_shap_values[0], last_X_test[0], 
                    feature_names=feature_names, matplotlib=True, show=False)
    plt.title(f"SHAP Force Plot for Building {last_building_id} (First Test Sample)")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'shap_force_plot_last_building_energy_sum_rf_F.png')
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
plt.title('Distribution of R² Scores Across Buildings (Random Forest)')
plt.tight_layout()
plot_path = os.path.join(output_dir, 'r2_distribution_energy_sum_rf_F.png')
plt.savefig(plot_path)
print(f"Saved R² Distribution plot: {plot_path}")
plt.show()
plt.close()

# RMSE Distribution
plt.figure(figsize=(8, 6))
plt.hist(rmse_scores, bins=10, color='lightgreen', edgecolor='black')
plt.xlabel('RMSE (kWh)')
plt.ylabel('Number of Buildings')
plt.title('Distribution of RMSE Across Buildings (Random Forest)')
plt.tight_layout()
plot_path = os.path.join(output_dir, 'rmse_distribution_energy_sum_rf_F.png')
plt.savefig(plot_path)
print(f"Saved RMSE Distribution plot: {plot_path}")
plt.show()
plt.close()

# Feature Importance
if feature_importances:
    mean_importance = np.mean(feature_importances, axis=0)
    feature_names = X_ind.columns
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, mean_importance, color='lightcoral')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Average Feature Importance Across Buildings (Random Forest)')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'feature_importance_energy_sum_rf_F.png')
    plt.savefig(plot_path)
    print(f"Saved Feature Importance plot: {plot_path}")
    plt.show()
    plt.close()

# Future Prediction
if 'future_predictions' in locals():
    plt.figure(figsize=(10, 6))
    plt.plot(future_dates, future_predictions, marker='o', color='green', label='Predicted Energy Sum')
    plt.xlabel('Date')
    plt.ylabel('Energy Sum (kWh)')
    plt.title(f'30-Day Energy Sum Prediction for Building {building_id} (Random Forest)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'future_prediction_energy_sum_rf.png')
    plt.savefig(plot_path)
    print(f"Saved Future Prediction plot: {plot_path}")
    plt.show()
    plt.close()

# --- Aggregate R² and RMSE Analysis ---
print("\n=== Aggregate R² and RMSE Analysis (Random Forest) ===")
if r2_scores and rmse_scores:
    print(f"Mean R²: {np.mean(r2_scores):.3f}")
    print(f"Min R²: {np.min(r2_scores):.3f}")
    print(f"Max R²: {np.max(r2_scores):.3f}")
    print(f"Median R²: {np.median(r2_scores):.3f}")
    print(f"Mean RMSE: {np.mean(rmse_scores):.4f}")
    print(f"Min RMSE: {np.min(rmse_scores):.4f}")
    print(f"Max RMSE: {np.max(rmse_scores):.4f}")
    print(f"Median RMSE: {np.median(rmse_scores):.4f}")
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