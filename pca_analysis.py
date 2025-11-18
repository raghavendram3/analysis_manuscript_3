import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

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
            # Convert numerical values, skip the label
            numerical_values = [float(x) for x in parts[1:]]
            data.append(numerical_values)

# Convert to numpy array
X = np.array(data)
print(f"Data shape: {X.shape}")
print(f"Number of samples: {len(labels)}")
print(f"Number of features: {X.shape[1]}\n")

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Print PCA results
print(f"\nExplained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative explained variance: {sum(pca.explained_variance_ratio_):.4f}")
print(f"PC1 explains {pca.explained_variance_ratio_[0]*100:.2f}% of variance")
print(f"PC2 explains {pca.explained_variance_ratio_[1]*100:.2f}% of variance\n")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: PCA scatter plot with labels
ax1 = axes[0, 0]
scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=range(len(labels)), 
                     cmap='viridis', s=100, alpha=0.6, edgecolors='black')
for i, label in enumerate(labels):
    ax1.annotate(label, (X_pca[i, 0], X_pca[i, 1]), 
                fontsize=8, alpha=0.7)
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
ax1.set_title('2D PCA Analysis - Samples Labeled')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='Sample Index')

# Plot 2: PCA scatter plot without labels (cleaner view)
ax2 = axes[0, 1]
scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=range(len(labels)), 
                      cmap='viridis', s=100, alpha=0.6, edgecolors='black')
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
ax2.set_title('2D PCA Analysis - Clean View')
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax2, label='Sample Index')

# Plot 3: Scree plot (variance explained)
ax3 = axes[1, 0]
# Compute PCA with all components for scree plot
pca_full = PCA()
pca_full.fit(X_scaled)
n_components_to_show = min(10, len(pca_full.explained_variance_ratio_))
ax3.bar(range(1, n_components_to_show + 1), 
       pca_full.explained_variance_ratio_[:n_components_to_show])
ax3.set_xlabel('Principal Component')
ax3.set_ylabel('Variance Explained Ratio')
ax3.set_title('Scree Plot - Variance Explained by Each PC')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Cumulative variance explained
ax4 = axes[1, 1]
cumsum_var = np.cumsum(pca_full.explained_variance_ratio_[:n_components_to_show])
ax4.plot(range(1, n_components_to_show + 1), cumsum_var, 
        marker='o', linestyle='-', linewidth=2, markersize=8)
ax4.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
ax4.set_xlabel('Number of Principal Components')
ax4.set_ylabel('Cumulative Variance Explained')
ax4.set_title('Cumulative Variance Explained')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_analysis_2d.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'pca_analysis_2d.png'")

# Create a detailed report
print("\n" + "="*60)
print("PCA ANALYSIS REPORT")
print("="*60)

# Feature contributions to PC1 and PC2
feature_names = header[1:]  # Skip 'Label' column
pc1_loadings = pca.components_[0]
pc2_loadings = pca.components_[1]

# Get top contributing features for PC1
pc1_contributions = sorted(zip(feature_names, pc1_loadings), 
                          key=lambda x: abs(x[1]), reverse=True)
print("\nTop 10 features contributing to PC1:")
for i, (feat, loading) in enumerate(pc1_contributions[:10], 1):
    print(f"{i}. {feat}: {loading:.4f}")

# Get top contributing features for PC2
pc2_contributions = sorted(zip(feature_names, pc2_loadings), 
                          key=lambda x: abs(x[1]), reverse=True)
print("\nTop 10 features contributing to PC2:")
for i, (feat, loading) in enumerate(pc2_contributions[:10], 1):
    print(f"{i}. {feat}: {loading:.4f}")

# Save PCA results to CSV
results_df = pd.DataFrame({
    'Label': labels,
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1]
})
results_df.to_csv('pca_results_2d.csv', index=False)
print("\nPCA coordinates saved to 'pca_results_2d.csv'")

# Save loadings to CSV
loadings_df = pd.DataFrame({
    'Feature': feature_names,
    'PC1_Loading': pc1_loadings,
    'PC2_Loading': pc2_loadings
})
loadings_df.to_csv('pca_loadings.csv', index=False)
print("Feature loadings saved to 'pca_loadings.csv'")

print("\nAnalysis complete!")