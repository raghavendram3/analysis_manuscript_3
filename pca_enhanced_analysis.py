import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns

# Load the dataset
data = pd.read_csv('data.txt', sep='\t')  # Adjust separator if necessary
features = data.columns[:-1]  # Assuming last column is the target/class

# Standardizing the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

# PCA Analysis
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data_scaled)
explained_variance = pca.explained_variance_ratio_

# Creating a DataFrame for PCA results
pca_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
pca_df['Cluster'] = None  # Placeholder for cluster assignments

# KMeans Clustering
kmeans = KMeans(n_clusters=3)  # Adjust n_clusters as needed
kmeans.fit(principalComponents)
pca_df['Cluster'] = kmeans.labels_

# Feature Loadings
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_df = pd.DataFrame(loadings, index=features, columns=['PC1', 'PC2'])

# Saving the loading information
loading_df.to_csv('feature_contributions.csv')

# Biplot Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis')
for i in range(loadings.shape[0]):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='r', alpha=0.5)
    plt.text(loadings[i, 0], loadings[i, 1], features[i], color='black', ha='center', va='center')
plt.title('PCA Biplot')
plt.savefig('biplot.png')
plt.close()

# Saving cluster assignments
pca_df.to_csv('cluster_assignments.csv', index=False)

# Final Visualization for PCA Results
plt.figure(figsize=(10, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Cluster'], cmap='viridis')
plt.title('PCA Enhanced Results')
plt.xlabel(f'PC1 ({explained_variance[0]:.2%} Variance)')
plt.ylabel(f'PC2 ({explained_variance[1]:.2%} Variance)')
plt.colorbar()
plt.savefig('pca_enhanced_results.png')
plt.close()

# Statistical Analysis of Clusters
stats = pca_df.groupby('Cluster').describe()
with open('cluster_analysis.txt', 'w') as f:
    f.write(str(stats))
