import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from itertools import product

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
epochs = 100
batch_size = 50

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

# --- Step 4: Grid Search for K-Means ---
print("Running Grid Search for K-Means...")
k_range = range(2, 11)  # Test K=2 to 10
kmeans_results = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_rbm)
    silhouette = silhouette_score(X_rbm, labels)
    cluster_sizes = np.bincount(labels)
    kmeans_results.append((k, silhouette, cluster_sizes))
    print(f"K={k}, Silhouette: {silhouette:.4f}, Cluster sizes: {cluster_sizes}")

# Find best K
best_k, best_silhouette, best_sizes = max(kmeans_results, key=lambda x: x[1])
print(f"\nBest K-Means: K={best_k}, Silhouette: {best_silhouette:.4f}, Cluster sizes: {best_sizes}")

# Run best K-Means
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans_labels = kmeans.fit_predict(X_rbm)

# Visualize best K-Means
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_rbm)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title(f'K-Means Clustering')
plt.grid(True)
plt.savefig(f'kmeans_clusters_1637_k{best_k}.png')
plt.show()

# K-Means profiles
building_agg['kmeans_cluster'] = kmeans_labels
kmeans_profiles = building_agg.groupby('kmeans_cluster')[features].mean()
print("K-Means Cluster Profiles:\n", kmeans_profiles)
kmeans_profiles.to_csv(f'kmeans_profiles_1637_k{best_k}.csv')

# --- Step 5: Grid Search for DBSCAN ---
print("Running Grid Search for DBSCAN...")
eps_range = [0.3, 0.4, 0.5, 0.6, 0.7]  # Distance thresholds
min_samples_range = [5, 10, 15, 20,50,100]  # Min points per cluster
dbscan_results = []

for eps, min_samples in product(eps_range, min_samples_range):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_rbm)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    if n_clusters > 1:  # Need 2+ clusters for silhouette
        clustered_indices = labels >= 0
        silhouette = silhouette_score(X_rbm[clustered_indices], labels[clustered_indices])
        cluster_sizes = np.bincount(labels[labels >= 0])
        dbscan_results.append((eps, min_samples, n_clusters, n_noise, silhouette, cluster_sizes))
        print(f"eps={eps}, min_samples={min_samples}, Clusters: {n_clusters}, Noise: {n_noise}, "
              f"Silhouette: {silhouette:.4f}, Sizes: {cluster_sizes}")
    else:
        print(f"eps={eps}, min_samples={min_samples}, Clusters: {n_clusters}, Noise: {n_noise}, "
              "Silhouette: N/A")

# Find best DBSCAN (max silhouette, reasonable noise)
if dbscan_results:
    best_eps, best_min_samples, best_n_clusters, best_n_noise, best_silhouette, best_sizes = max(
        dbscan_results, key=lambda x: x[4] if x[3] / 1637 < 0.2 else -1)  # Prefer <20% noise
    print(f"\nBest DBSCAN: eps={best_eps}, min_samples={best_min_samples}, Clusters: {best_n_clusters}, "
          f"Noise: {best_n_noise}, Silhouette: {best_silhouette:.4f}, Sizes: {best_sizes}")
else:
    print("No valid DBSCAN results with >1 cluster")

# Run best DBSCAN
if dbscan_results:
    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    dbscan_labels = dbscan.fit_predict(X_rbm)

    # Visualize best DBSCAN
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster (-1 = Noise)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'DBSCAN Clustering )')
    plt.grid(True)
    plt.savefig(f'dbscan_clusters_1637_eps{best_eps}_min{best_min_samples}.png')
    plt.show()

    # DBSCAN profiles
    building_agg['dbscan_cluster'] = dbscan_labels
    dbscan_profiles = building_agg[building_agg['dbscan_cluster'] >= 0].groupby('dbscan_cluster')[features].mean()
    print("DBSCAN Cluster Profiles:\n", dbscan_profiles)
    dbscan_profiles.to_csv(f'dbscan_profiles_1637_eps{best_eps}_min{best_min_samples}.csv')