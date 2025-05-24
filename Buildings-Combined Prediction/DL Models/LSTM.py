import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Step 1: Load and Select Building
df = pd.read_csv(r"C:\Users\ritha\OneDrive\Desktop\2ND SEMESTER\EEE\UK\Merged_Dataset.csv")
df["day"] = pd.to_datetime(df["day"], dayfirst=True)

unique_buildings = df["building_id"].unique()
print("Total buildings:", len(unique_buildings))
print("Sample building IDs:", unique_buildings[:5], "...")

while True:
    try:
        building_id = int(input("Enter a building ID (0-1636) for individual analysis: "))
        if building_id in unique_buildings:
            break
        else:
            print(f"Error: '{building_id}' not found. Enter a number between 0 and 1636.")
    except ValueError:
        print("Error: Enter a valid integer building ID.")

# Filter data for individual building
features = ['day', 'energy_sum', 'energy_mean', 'temperatureMax', 'humidity', 'windSpeed', 'cloudCover', 'uvIndex']
building_data = df[df["building_id"] == building_id][features]
daily_data = building_data.set_index('day').resample('D').mean().ffill()

# Explore individual building
print("\nIndividual Building Analysis:")
print("Building ID:", building_id)
print("Raw daily energy_sum (first 5):", daily_data['energy_sum'].head().values)
print("Raw daily energy_sum stats:", daily_data['energy_sum'].describe())
print("Total days:", len(daily_data))

# Step 2: Scale Data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(daily_data)
print("Scaled data (first 5):", scaled_data[:5])

# Step 3: Build Sequences
lookback = 20
X, y = [], []
for i in range(lookback, len(scaled_data)):
    X.append(scaled_data[i - lookback:i])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)
print("X shape:", X.shape)
print("y shape:", y.shape)

# Step 4: Split Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Step 5: Build Stacked LSTM Model with Bidirectional Layer
model = Sequential()
model.add(Bidirectional(LSTM(100, return_sequences=True, kernel_regularizer=l2(0.005), recurrent_regularizer=l2(0.005)), 
                        input_shape=(lookback, 7)))
model.add(Dropout(0.2))
model.add(LSTM(100, kernel_regularizer=l2(0.005), recurrent_regularizer=l2(0.005)))
model.add(Dropout(0.2))
model.add(Dense(20, activation='tanh', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse')

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00005)

model.summary()
print("Training for 500 epochs...")
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), 
                    callbacks=[reduce_lr], verbose=1)

# Plot training loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title(f"Training Loss for Building {building_id} (500 Epochs)")
plt.show()

# Step 6: Predict and Compare
y_pred_scaled = model.predict(X_test)
energy_max = daily_data['energy_sum'].max()
energy_min = daily_data['energy_sum'].min()
y_pred_unscaled = y_pred_scaled * (energy_max - energy_min) + energy_min
y_test_unscaled = y_test * (energy_max - energy_min) + energy_min

r2 = r2_score(y_test_unscaled, y_pred_unscaled)
y_mean = np.full_like(y_test_unscaled, y_train.mean() * (energy_max - energy_min) + energy_min)
r2_baseline = r2_score(y_test_unscaled, y_mean)

plt.plot(y_pred_unscaled, label='Predicted Energy', color='orange', linestyle='-', marker='o')
plt.xlabel('Test Day')
plt.ylabel('Energy (unscaled)')
plt.title(f"Predicted Energy for Building {building_id} (R² = {r2:.3f})")
plt.legend()
plt.show()

print("Last 5 real (unscaled):", y_test_unscaled[-5:])
print("Last 5 predicted (unscaled):", y_pred_unscaled[-5:])
print(f"R² Score (Model): {r2:.3f}")
print(f"R² Score (Mean Baseline): {r2_baseline:.3f}")