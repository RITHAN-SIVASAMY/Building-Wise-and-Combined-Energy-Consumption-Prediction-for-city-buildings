import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Step 4: Build and Train LSTM Model with PyTorch

# Load reduced dataset
print("Loading reduced dataset 'reduced_energy_data.csv'...")
df = pd.read_csv('reduced_energy_data.csv')
print(f"Dataset loaded. Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Verify key columns
required_cols = ['energy_sum', 'energy_mean', 'building_id']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"Error: Missing columns: {missing_cols}")
    raise ValueError("Required columns not found in dataset.")
print("Confirmed: 'energy_sum', 'energy_mean', and 'building_id' present.")

# Diagnose data
print("\nData Diagnostics:")
print(f"building_id dtype: {df['building_id'].dtype}")
print(f"Sample building_id values: {df['building_id'].head().tolist()}")
print(f"energy_sum stats:\n{df['energy_sum'].describe()}")
print(f"energy_mean stats:\n{df['energy_mean'].describe()}")
print(f"Number of unique building_id: {df['building_id'].nunique()}")

# Check for chronological order
if 'day_of_week' in df.columns:
    is_sorted = df['day_of_week'].is_monotonic_increasing
    print(f"Data sorted by day_of_week: {is_sorted}")
    if not is_sorted:
        print("Sorting data by day_of_week, month, day_of_month...")
        df.sort_values(['day_of_week', 'month', 'day_of_month'], inplace=True)
else:
    print("No day_of_week column, assuming chronological order...")

# Define features and target
target = 'energy_sum'
features = [col for col in df.columns if col != target]
print(f"Features: {features}")
print(f"Target: {target}")

# Normalize features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
if df['building_id'].dtype == 'object' or df['building_id'].dtype == 'string':
    print("Treating 'building_id' as categorical, excluding from scaling...")
    scale_cols = [col for col in features if col != 'building_id']
    X_scaled = df[scale_cols].copy()
    X_scaled[scale_cols] = scaler_X.fit_transform(X_scaled[scale_cols])
    X_scaled['building_id'] = df['building_id']
else:
    print("Treating 'building_id' as numerical, including in scaling...")
    X_scaled = df[features].copy()
    X_scaled[features] = scaler_X.fit_transform(X_scaled[features])
y_scaled = scaler_y.fit_transform(df[[target]])
print(f"Scaled feature matrix shape: {X_scaled.shape}")
print(f"Scaled target shape: {y_scaled.shape}")

# Convert to numpy arrays
X = X_scaled[features].values
y = y_scaled  # Use scaled target
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Create sequences
def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i + seq_length])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)

seq_length = 14  # Increased from 7
print(f"Creating sequences with length {seq_length}...")
X_seq, y_seq = create_sequences(X, y, seq_length)
print(f"Sequence X shape: {X_seq.shape}")
print(f"Sequence y shape: {y_seq.shape}")

# Split into train, validation, and test sets
train_size = int(0.6 * len(X_seq))
val_size = int(0.2 * len(X_seq))
X_train = X_seq[:train_size]
y_train = y_seq[:train_size]
X_val = X_seq[train_size:train_size + val_size]
y_val = y_seq[train_size:train_size + val_size]
X_test = X_seq[train_size + val_size:]
y_test = y_seq[train_size + val_size:]
print(f"Train shape: {X_train.shape}, {y_train.shape}")
print(f"Validation shape: {X_val.shape}, {y_val.shape}")
print(f"Test shape: {X_test.shape}, {y_test.shape}")

# Custom Dataset
class EnergyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create DataLoaders
batch_size = 32
train_dataset = EnergyDataset(X_train, y_train)
val_dataset = EnergyDataset(X_val, y_val)
test_dataset = EnergyDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"DataLoaders created with batch size {batch_size}.")

# Define Stacked Bidirectional LSTM model
class LSTMEnergyModel(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, dropout=0.2):
        super(LSTMEnergyModel, self).__init__()
        # First bidirectional LSTM layer
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)
        # Second bidirectional LSTM layer
        self.lstm2 = nn.LSTM(hidden_size1 * 2, hidden_size2, batch_first=True, bidirectional=True)  # *2 for bidirectional
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size2 * 2, 32)  # *2 for bidirectional output
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = out[:, -1, :]  # Take the last time step
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Initialize model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = LSTMEnergyModel(input_size=X_train.shape[2]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Lowered lr
print("LSTM model initialized.")

# Training loop with detailed progress
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100, patience=15):
    train_losses = []
    val_losses = []
    train_r2s = []
    val_r2s = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        batch_losses = []
        train_preds = []
        train_true = []
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("Training Batches:")
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch.squeeze())
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            train_loss += batch_loss * X_batch.size(0)
            train_preds.extend(outputs.squeeze().cpu().detach().numpy())
            train_true.extend(y_batch.squeeze().cpu().numpy())

            # Print batch progress
            if (batch_idx + 1) % 1000 == 0 or (batch_idx + 1) == len(train_loader):
                running_avg_loss = np.mean(batch_losses[-1000:]) if len(batch_losses) >= 1000 else np.mean(batch_losses)
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, "
                      f"Batch Loss: {batch_loss:.6f}, "
                      f"Running Avg Loss: {running_avg_loss:.6f}")

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        train_r2 = r2_score(train_true, train_preds)
        train_r2s.append(train_r2)
        print(f"Epoch {epoch + 1} Training Loss: {train_loss:.6f}, Training R²: {train_r2:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch.squeeze())
                val_loss += loss.item() * X_batch.size(0)
                val_preds.extend(outputs.squeeze().cpu().numpy())
                val_true.extend(y_batch.squeeze().cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        val_r2 = r2_score(val_true, val_preds)
        val_r2s.append(val_r2)
        print(f"Epoch {epoch + 1} Validation Loss: {val_loss:.6f}, Validation R²: {val_r2:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            print("  New best validation loss, saving model state.")
        else:
            patience_counter += 1
            print(f"  Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Restore best model
    model.load_state_dict(best_model_state)
    return train_losses, val_losses, train_r2s, val_r2s

# Train model
print("Training LSTM model...")
train_losses, val_losses, train_r2s, val_r2s = train_model(
    model, train_loader, val_loader, criterion, optimizer, epochs=100, patience=15
)
print("Training completed.")

# Evaluate on test set
print("Evaluating model on test set...")
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        y_pred.extend(outputs.squeeze().cpu().numpy())
        y_true.extend(y_batch.squeeze().cpu().numpy())
y_pred = np.array(y_pred)
y_true = np.array(y_true)

# Inverse transform predictions and true values
y_pred_unscaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_true_unscaled = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()

rmse = np.sqrt(mean_squared_error(y_true_unscaled, y_pred_unscaled))
mae = mean_absolute_error(y_true_unscaled, y_pred_unscaled)
r2 = r2_score(y_true_unscaled, y_pred_unscaled)
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R²: {r2:.4f}")

# Save model
print("Saving trained model...")
torch.save(model.state_dict(), 'New folder/lstm_energy_model.pt')
print("Model saved to 'lstm_energy_model.pt'.")

# Save predictions
print("Saving test predictions...")
pred_df = pd.DataFrame({
    'actual_energy_sum': y_true_unscaled,
    'predicted_energy_sum': y_pred_unscaled
})
pred_df.to_csv('lstm_predictions.csv', index=False)
print("Predictions saved to 'lstm_predictions.csv'.")

# Plot training history
print("Plotting training history...")
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('LSTM Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.savefig('lstm_training_history.png')
plt.close()
print("Training history plot saved to 'lstm_training_history.png'.")

# Plot R² history
print("Plotting R² history...")
plt.figure(figsize=(10, 6))
plt.plot(train_r2s, label='Training R²')
plt.plot(val_r2s, label='Validation R²')
plt.title('LSTM R² History')
plt.xlabel('Epoch')
plt.ylabel('R² Score')
plt.legend()
plt.savefig('lstm_r2_history.png')
plt.close()
print("R² history plot saved to 'lstm_r2_history.png'.")

# Plot actual vs predicted
print("Plotting actual vs predicted...")
plt.figure(figsize=(10, 6))
plt.scatter(y_true_unscaled, y_pred_unscaled, alpha=0.5, s=10)
plt.plot([y_true_unscaled.min(), y_true_unscaled.max()], [y_true_unscaled.min(), y_true_unscaled.max()], 'r--')
plt.title('Actual vs Predicted Energy Sum')
plt.xlabel('Actual energy_sum')
plt.ylabel('Predicted energy_sum')
plt.savefig('actual_vs_predicted.png')
plt.close()
print("Actual vs predicted plot saved to 'actual_vs_predicted.png'.")