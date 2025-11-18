import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import pandas as pd
import seaborn as sns

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

print("="*70)
print("ENHANCED PCA ANALYSIS WITH CLUSTERING AND FEATURE CONTRIBUTIONS")
print("="*70)

# Load the data
data = []
labels = []
header = None

with open('data.txt', 'r') as f:
    lines = f.readlines()
    header = lines[0].strip().split('\t')
    
    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) > 1:
            labels.append(parts[0])
            numerical_values = [float(x) for x in parts[1:]]
            data.append(numerical_values)

# Convert to numpy array
X = np.array(data)
feature_names = header[1:]
print(f"\nData loaded: {X.shape[0]} samples, {X.shape[1]} features")

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA Results:")
print(f"  PC1 explains: {pca.explained_variance_ratio_[0]*100:.2f}% of variance")
print(f"  PC2 explains: {pca.explained_variance_ratio_[1]*100:.2f}% of variance")
print(f"  Total explained: {sum(pca.explained_variance_ratio_)*100:.2f}%")

# ============================================================================
# 1. FEATURE CONTRIBUTIONS ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("FEATURE CONTRIBUTIONS TO PRINCIPAL COMPONENTS")
print("="*70)

# Get loadings (eigenvectors scaled by sqrt of eigenvalues)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Create DataFrame for loadings
loadings_df = pd.DataFrame(
    loadings,
    columns=['PC1_loading', 'PC2_loading'],
    index=feature_names
)
loadings_df['PC1_abs'] = np.abs(loadings_df['PC1_loading'])
loadings_df['PC2_abs'] = np.abs(loadings_df['PC2_loading'])

# Sort and display top contributors
print("\nTop 10 features contributing to PC1:")
top_pc1 = loadings_df.nlargest(10, 'PC1_abs')[['PC1_loading', 'PC1_abs']]
for i, (idx, row) in enumerate(top_pc1.iterrows(), 1):
    print(f"  {i}. {idx:30s}: {row['PC1_loading']:7.4f}")

print("\nTop 10 features contributing to PC2:")
top_pc2 = loadings_df.nlargest(10, 'PC2_abs')[['PC2_loading', 'PC2_abs']]
for i, (idx, row) in enumerate(top_pc2.iterrows(), 1):
    print(f"  {i}. {idx:30s}: {row['PC2_loading']:7.4f}")

# Save full loadings
loadings_df.to_csv('feature_loadings_detailed.csv')
print("\nFull feature loadings saved to 'feature_loadings_detailed.csv'")

# ============================================================================
# 2. OPTIMAL CLUSTER ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("K-MEANS CLUSTERING ANALYSIS")
print("="*70)

# Determine optimal number of clusters using elbow method
distortions = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    distortions.append(sum(np.min(cdist(X_pca, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X_pca.shape[0])

# Use 4 clusters as default
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_pca)

print(f"\nUsing {optimal_k} clusters")
print("\nCluster distribution:")
for i in range(optimal_k):
    cluster_samples = [labels[j] for j in range(len(labels)) if cluster_labels[j] == i]
    print(f"\nCluster {i} ({len(cluster_samples)} samples):")
    print(f"  Samples: {', '.join(cluster_samples)}")

# ============================================================================
# 3. COMPREHENSIVE VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# Create comprehensive figure
fig = plt.figure(figsize=(20, 12))

# Plot 1: Biplot with top features
ax1 = plt.subplot(2, 3, 1)
scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                     cmap='viridis', s=100, alpha=0.6, edgecolors='black', linewidth=1.5)

# Add sample labels
for i, label in enumerate(labels):
    ax1.annotate(label, (X_pca[i, 0], X_pca[i, 1]), 
                fontsize=7, alpha=0.7, ha='center')

# Add feature vectors (top 8 most important)
scale_factor = 3.5
top_features_pc1 = loadings_df.nlargest(4, 'PC1_abs').index.tolist()
top_features_pc2 = loadings_df.nlargest(4, 'PC2_abs').index.tolist()
top_features = list(set(top_features_pc1 + top_features_pc2))

for feature in top_features:
    idx = feature_names.index(feature)
    ax1.arrow(0, 0, loadings[idx, 0]*scale_factor, loadings[idx, 1]*scale_factor,
             head_width=0.3, head_length=0.2, fc='red', ec='red', alpha=0.7, linewidth=2)
    ax1.text(loadings[idx, 0]*scale_factor*1.15, loadings[idx, 1]*scale_factor*1.15,
            feature, fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)', fontsize=12)
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)', fontsize=12)
ax1.set_title('PCA Biplot with Top Feature Vectors', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='Cluster')

# Plot 2: Clean cluster view
ax2 = plt.subplot(2, 3, 2)
scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                      cmap='viridis', s=150, alpha=0.7, edgecolors='black', linewidth=1.5)
# Add cluster centers
centers = kmeans.cluster_centers_
ax2.scatter(centers[:, 0], centers[:, 1], c='red', s=300, alpha=0.8, 
           edgecolors='black', linewidth=2, marker='*', label='Centroids')
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)', fontsize=12)
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)', fontsize=12)
ax2.set_title('K-Means Clustering Results', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax2, label='Cluster')

# Plot 3: Elbow plot
ax3 = plt.subplot(2, 3, 3)
ax3.plot(K_range, distortions, 'bo-', linewidth=2, markersize=8)
ax3.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'Selected k={optimal_k}')
ax3.set_xlabel('Number of Clusters (k)', fontsize=12)
ax3.set_ylabel('Average Distortion', fontsize=12)
ax3.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: PC1 Loading bar chart
ax4 = plt.subplot(2, 3, 4)
top10_pc1 = loadings_df.nlargest(10, 'PC1_abs')
colors = ['green' if x > 0 else 'red' for x in top10_pc1['PC1_loading']]
ax4.barh(range(10), top10_pc1['PC1_loading'], color=colors, alpha=0.7, edgecolor='black')
ax4.set_yticks(range(10))
ax4.set_yticklabels(top10_pc1.index, fontsize=9)
ax4.set_xlabel('Loading Value', fontsize=12)
ax4.set_title('Top 10 Features for PC1', fontsize=14, fontweight='bold')
ax4.axvline(x=0, color='black', linewidth=1)
ax4.grid(True, alpha=0.3, axis='x')

# Plot 5: PC2 Loading bar chart
ax5 = plt.subplot(2, 3, 5)
top10_pc2 = loadings_df.nlargest(10, 'PC2_abs')
colors = ['green' if x > 0 else 'red' for x in top10_pc2['PC2_loading']]
ax5.barh(range(10), top10_pc2['PC2_loading'], color=colors, alpha=0.7, edgecolor='black')
ax5.set_yticks(range(10))
ax5.set_yticklabels(top10_pc2.index, fontsize=9)
ax5.set_xlabel('Loading Value', fontsize=12)
ax5.set_title('Top 10 Features for PC2', fontsize=14, fontweight='bold')
ax5.axvline(x=0, color='black', linewidth=1)
ax5.grid(True, alpha=0.3, axis='x')

# Plot 6: Cluster statistics heatmap
ax6 = plt.subplot(2, 3, 6)
cluster_stats = []
for i in range(optimal_k):
    mask = cluster_labels == i
    cluster_stats.append([
        np.mean(X_pca[mask, 0]),
        np.mean(X_pca[mask, 1]),
        np.std(X_pca[mask, 0]),
        np.std(X_pca[mask, 1])
    ])
cluster_stats_df = pd.DataFrame(cluster_stats,  
                               columns=['PC1_mean', 'PC2_mean', 'PC1_std', 'PC2_std'],
                               index=[f'Cluster {i}' for i in range(optimal_k)])
sns.heatmap(cluster_stats_df, annot=True, fmt='.2f', cmap='coolwarm',  
           center=0, ax=ax6, cbar_kws={'label': 'Value'})
ax6.set_title('Cluster Statistics', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('pca_enhanced_analysis.png', dpi=300, bbox_inches='tight')
print("\nComprehensive visualization saved as 'pca_enhanced_analysis.png'")

# ============================================================================
# 4. DETAILED CLUSTER ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("DETAILED CLUSTER STATISTICS")
print("="*70)

# Create detailed cluster analysis report
cluster_report = []
for i in range(optimal_k):
    mask = cluster_labels == i
    cluster_samples = [labels[j] for j in range(len(labels)) if cluster_labels[j] == i]
    
    print(f"\n{'='*70}")
    print(f"CLUSTER {i} - {len(cluster_samples)} samples")
    print(f"{'='*70}")
    print(f"Samples: {', '.join(cluster_samples)}")
    print(f"\nPCA Coordinates:")
    print(f"  PC1: mean={np.mean(X_pca[mask, 0]):.3f}, std={np.std(X_pca[mask, 0]):.3f}")
    print(f"  PC2: mean={np.mean(X_pca[mask, 1]):.3f}, std={np.std(X_pca[mask, 1]):.3f}")
    
    # Feature statistics for this cluster
    cluster_data = X[mask]
    feature_means = np.mean(cluster_data, axis=0)
    
    # Find top 5 distinguishing features
    sorted_indices = np.argsort(feature_means)
    print(f"\nTop 5 features with LOWEST values:")
    for idx in sorted_indices[:5]:
        print(f"  {feature_names[idx]:30s}: {feature_means[idx]:.4f}")
    
    print(f"\nTop 5 features with HIGHEST values:")
    for idx in sorted_indices[-5:][::-1]:
        print(f"  {feature_names[idx]:30s}: {feature_means[idx]:.4f}")
    
    cluster_report.append({
        'Cluster': i,
        'N_samples': len(cluster_samples),
        'Samples': ', '.join(cluster_samples),
        'PC1_mean': np.mean(X_pca[mask, 0]),
        'PC1_std': np.std(X_pca[mask, 0]),
        'PC2_mean': np.mean(X_pca[mask, 1]),
        'PC2_std': np.std(X_pca[mask, 1])
    })

# Save results
results_df = pd.DataFrame({
    'Label': labels,
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'Cluster': cluster_labels
})
results_df.to_csv('pca_results_with_clusters.csv', index=False)

cluster_report_df = pd.DataFrame(cluster_report)
cluster_report_df.to_csv('cluster_summary.csv', index=False)

print("\n" + "="*70)
print("FILES SAVED:")
print("="*70)
print("  1. pca_enhanced_analysis.png - Comprehensive visualization")
print("  2. pca_results_with_clusters.csv - PCA coordinates with cluster assignments")
print("  3. cluster_summary.csv - Statistical summary of each cluster")
print("  4. feature_loadings_detailed.csv - Feature contributions to PCs")
print("\nAnalysis complete!")
print("="*70)
