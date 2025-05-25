import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Step 1: Preprocess Full Dataset ---
print("Loading and preprocessing data...")
file_path = r"C:\Users\ritha\OneDrive\Desktop\2ND SEMESTER\EEE\UK\Merged_Dataset.csv"
df = pd.read_csv(file_path)

# Parse 'day'
df['day'] = pd.to_datetime(df['day'], dayfirst=True)

# Fill missing values
df.ffill(inplace=True)
df.bfill(inplace=True)

# Features
features = ['energy_sum', 'energy_mean', 'energy_median', 'temperatureMax', 
            'humidity', 'windSpeed', 'cloudCover', 'uvIndex']
print(f"Dataset size: {df.shape}, Buildings: {len(df['building_id'].unique())}")

# --- Step 2: Aggregate Daily Data ---
print("Aggregating daily data...")
daily_agg = df.groupby(['building_id', 'day'])[features].mean().reset_index()
building_agg = daily_agg.groupby('building_id')[features].mean()
print(f"Aggregated shape: {building_agg.shape}")
X = building_agg.values

# --- Step 3: Train RBM ---
print("Training RBM...")
class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.1)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))

    def forward(self, v):
        h_prob = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)
        h_sample = torch.bernoulli(h_prob)
        return h_sample, h_prob

    def reconstruct(self, h):
        v_prob = torch.sigmoid(torch.matmul(h, self.W) + self.v_bias)
        return v_prob

# RBM setup
n_visible = X.shape[1]  # 8 features
n_hidden = 10  # 10 latent features
rbm = RBM(n_visible, n_hidden).to(device)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

# Train RBM
optimizer = torch.optim.Adam(rbm.parameters(), lr=0.01)
epochs = 50
batch_size = 100

for epoch in range(epochs):
    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i:i + batch_size]
        optimizer.zero_grad()
        h_sample, h_prob = rbm(batch)
        v_recon = rbm.reconstruct(h_sample)
        loss = torch.mean(torch.sum((batch - v_recon) ** 2, dim=1))
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Extract features
with torch.no_grad():
    _, h_prob = rbm(X_tensor)
    X_rbm = h_prob.cpu().numpy()
print(f"RBM features shape: {X_rbm.shape}")

# --- Step 4: Grid Search for GMM ---
print("Running Grid Search for GMM...")
k_range = range(2, 11)  # Test K=2 to 10
gmm_results = []

for k in k_range:
    gmm = GaussianMixture(n_components=k, random_state=42)
    labels = gmm.fit_predict(X_rbm)
    silhouette = silhouette_score(X_rbm, labels)
    cluster_sizes = np.bincount(labels)
    bic = gmm.bic(X_rbm)  # Bayesian Information Criterion
    gmm_results.append((k, silhouette, cluster_sizes, bic))
    print(f"K={k}, Silhouette: {silhouette:.4f}, Cluster sizes: {cluster_sizes}, BIC: {bic:.4f}")

# Find best K (highest silhouette, or balance with BIC)
best_k, best_silhouette, best_sizes, best_bic = max(gmm_results, key=lambda x: x[1])
print(f"\nBest GMM: K={best_k}, Silhouette: {best_silhouette:.4f}, Cluster sizes: {best_sizes}, BIC: {best_bic:.4f}")

# Run best GMM
gmm = GaussianMixture(n_components=best_k, random_state=42)
gmm_labels = gmm.fit_predict(X_rbm)
gmm_probs = gmm.predict_proba(X_rbm)  # Probability per cluster

# Visualize best GMM
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_rbm)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title(f'GMM Clustering)')
plt.grid(True)
plt.savefig(f'gmm_clusters_1637_k{best_k}.png')
plt.show()

# GMM profiles with probabilities
building_agg['gmm_cluster'] = gmm_labels
gmm_profiles = building_agg.groupby('gmm_cluster')[features].mean()
print("GMM Cluster Profiles:\n", gmm_profiles)
gmm_profiles.to_csv(f'gmm_profiles_1637_k{best_k}.csv')

# Save probabilities
probs_df = pd.DataFrame(gmm_probs, columns=[f'Cluster_{i}' for i in range(best_k)])
probs_df['building_id'] = building_agg.index
probs_df.to_csv(f'gmm_probabilities_1637_k{best_k}.csv', index=False)
print("GMM Probabilities saved to file.")

# Silhouette for best GMM (redundant but explicit)
sil_score_gmm = silhouette_score(X_rbm, gmm_labels)
print(f"Best GMM Silhouette Score: {sil_score_gmm:.4f}")