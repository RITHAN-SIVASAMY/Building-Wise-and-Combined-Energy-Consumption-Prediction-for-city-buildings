import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seed for reproducibility
print("Setting random seeds...")
torch.manual_seed(42)
np.random.seed(42)
print("Random seeds set.")

# Load dataset
print("Loading dataset...")
data = pd.read_csv('Filtered_Dataset.csv')
print(f"Dataset loaded with {len(data)} rows and {len(data.columns)} columns.")

# Convert day to datetime
print("Parsing dates in 'day' column...")
try:
    data['day'] = pd.to_datetime(data['day'], format='mixed', dayfirst=True, errors='coerce')
except ValueError as e:
    print(f"Error parsing dates: {e}")
    print("Unique date values:", data['day'].unique())
    raise
print("Date parsing completed.")

# Check for NaT values
if data['day'].isna().any():
    print("Warning: Some dates could not be parsed and are NaT")
    print(data[data['day'].isna()])
    data = data.dropna(subset=['day'])
    print(f"Dropped rows with NaT dates. Remaining rows: {len(data)}")

# Analyze target distribution
print("Analyzing target distribution...")
print(f"Energy_sum statistics:\n{data['energy_sum'].describe()}")
print(f"Number of zero energy_sum values: {(data['energy_sum'] == 0).sum()}")
print(f"Skewness of energy_sum: {data['energy_sum'].skew():.4f}")

# Preprocess target (log-transform to handle skewness)
print("Applying log-transformation to energy_sum...")
data['energy_sum_log'] = np.log1p(data['energy_sum'])  # log(1+x) to handle zeros

# Add temporal and lagged features
print("Adding temporal and lagged features...")
data['day_of_week'] = data['day'].dt.dayofweek
data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
data['month'] = data['day'].dt.month
# Add lagged energy_sum (previous day)
data = data.sort_values(['building_id', 'day'])
data['energy_sum_lag1'] = data.groupby('building_id')['energy_sum'].shift(1)
data = data.dropna().reset_index(drop=True)  # Drop rows with NaN lags

# Features to use
features = ['energy_mean', 'energy_max', 'energy_min', 'temperatureMax', 'temperatureMin',
            'day_of_week', 'is_weekend', 'month', 'energy_sum_lag1']
target = 'energy_sum_log'


# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Function to create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length][-1])  # target is last column
    return np.array(X), np.array(y)


# Model parameters
seq_length = 14  # Increased to capture more context
hidden_size = 16  # Further reduced
num_layers = 1
num_epochs = 100
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Store metrics
results = {'building_id': [], 'r2': [], 'mse': [], 'data_points': [], 'sequences': [],
           'target_variance': [], 'baseline_r2': []}
negative_r2_buildings = []

# Summarize data distribution
print("Summarizing data distribution across buildings...")
building_counts = data.groupby('building_id').size()
print(f"Data points per building:\n{building_counts.describe()}")

# Process each building
building_ids = data['building_id'].unique()
print(f"Processing {len(building_ids)} buildings...")
for building_id in tqdm(building_ids, desc="Training models", unit="building"):
    # Get building data
    building_data = data[data['building_id'] == building_id].sort_values('day')
    print(f"\nBuilding {building_id}: {len(building_data)} data points")

    # Check target variance
    target_var = building_data[target].var()
    print(f"Building {building_id}: Target variance = {target_var:.6f}")

    # Prepare features and target
    X = building_data[features].values
    y = building_data[target].values

    # Scale features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Combine scaled features and target
    data_scaled = np.hstack((X_scaled, y_scaled.reshape(-1, 1)))

    # Create sequences
    X_seq, y_seq = create_sequences(data_scaled, seq_length)
    print(f"Building {building_id}: Created {len(X_seq)} sequences")

    # Skip buildings with too few sequences
    if len(X_seq) < 50:
        print(f"Skipping building {building_id}: too few sequences ({len(X_seq)})")
        continue

    # Train-test split (70-30)
    train_size = int(0.7 * len(X_seq))
    if train_size == 0 or len(X_seq) - train_size == 0:
        print(f"Skipping building {building_id}: insufficient data for train-test split")
        continue
    X_train = X_seq[:train_size]
    y_train = y_seq[:train_size]
    X_test = X_seq[train_size:]
    y_test = y_seq[train_size:]
    print(f"Building {building_id}: Train = {len(X_train)}, Test = {len(X_test)}")

    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)

    # Initialize model
    model = LSTMModel(input_size=len(features) + 1, hidden_size=hidden_size, num_layers=num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    model.train()
    for epoch in range(num_epochs):
        outputs = model(X_train)
        optimizer.zero_grad()
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()

    # Evaluate model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).squeeze()
        y_pred = y_pred.cpu().numpy()
        y_test_np = y_test.cpu().numpy()

        # Inverse transform predictions and actual values
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_test_np = scaler_y.inverse_transform(y_test_np.reshape(-1, 1)).flatten()

        # Transform back to original scale
        y_pred_orig = np.expm1(y_pred)  # Inverse of log1p
        y_test_orig = np.expm1(y_test_np)

        # Calculate metrics
        r2 = r2_score(y_test_orig, y_pred_orig)
        mse = mean_squared_error(y_test_orig, y_pred_orig)

        # Calculate baseline R² (mean predictor)
        baseline_pred = np.full_like(y_test_orig, y_test_orig.mean())
        baseline_r2 = r2_score(y_test_orig, baseline_pred)

        # Store results
        results['building_id'].append(building_id)
        results['r2'].append(r2)
        results['mse'].append(mse)
        results['data_points'].append(len(building_data))
        results['sequences'].append(len(X_seq))
        results['target_variance'].append(target_var)
        results['baseline_r2'].append(baseline_r2)

        print(f"Building {building_id}: R² = {r2:.4f}, MSE = {mse:.6f}, Baseline R² = {baseline_r2:.4f}")
        if r2 < 0:
            negative_r2_buildings.append(building_id)
            print(f"Building {building_id}: Negative R² detected")

# Create DataFrame with results
results_df = pd.DataFrame(results)
print("\nResults for each building:")
print(results_df)
print(f"\nNumber of buildings with negative R²: {len(negative_r2_buildings)}")
print(f"Buildings with negative R²: {negative_r2_buildings}")
results_df.to_csv('lstm_results.csv', index=False)
print("Results saved to 'lstm_results.csv'")

# Plot R² distribution
print("Generating R² distribution plot...")
plt.figure(figsize=(10, 6))
plt.hist(results_df['r2'], bins=30, edgecolor='black')
plt.title('Distribution of R² Scores Across Buildings')
plt.xlabel('R² Score')
plt.ylabel('Number of Buildings')
plt.grid(True)
plt.savefig('r2_distribution.png')
plt.show()
print("Plot saved as 'r2_distribution.png'")

# Summary of negative R² buildings
if negative_r2_buildings:
    print("\nSummary of buildings with negative R²:")
    neg_r2_df = results_df[results_df['building_id'].isin(negative_r2_buildings)]
    print(neg_r2_df.describe())

# Plot actual vs. predicted for up to 5 buildings with most negative R²
if negative_r2_buildings:
    print("\nGenerating diagnostic plots for buildings with most negative R²...")
    worst_buildings = results_df[results_df['building_id'].isin(negative_r2_buildings)][
        ['building_id', 'r2']].sort_values('r2').head(5)
    for _, row in worst_buildings.iterrows():
        building_id = int(row['building_id'])
        building_data = data[data['building_id'] == building_id].sort_values('day')
        X = building_data[features].values
        y = building_data[target].values
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        data_scaled = np.hstack((X_scaled, y_scaled.reshape(-1, 1)))
        X_seq, y_seq = create_sequences(data_scaled, seq_length)
        test_size = int(0.3 * len(X_seq))
        X_test = X_seq[-test_size:]
        y_test = y_seq[-test_size:]
        X_test = torch.FloatTensor(X_test).to(device)
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test).squeeze().cpu().numpy()
            y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_orig = np.expm1(y_pred)
            y_test_orig = np.expm1(y_test)

        plt.figure(figsize=(10, 6))
        plt.plot(y_test_orig, label='Actual')
        plt.plot(y_pred_orig, label='Predicted')
        plt.title(f'Building {building_id}: Actual vs Predicted (R² = {row["r2"]:.4f})')
        plt.xlabel('Test Sample Index')
        plt.ylabel('Energy Sum')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'building_{building_id}_actual_vs_predicted.png')
        plt.show()
        print(f"Saved diagnostic plot for building {building_id}")