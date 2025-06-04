import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('sample_openpowerlifting.csv')

numeric_cols = ['Age', 'BodyweightKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'TotalKg']
X = df[numeric_cols].dropna()

linked_raw = linkage(X, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(linked_raw, truncate_mode='lastp', p=30, leaf_rotation=90, leaf_font_size=10)
plt.title("Hierarchical Clustering Dendrogram (Raw Data)")
plt.tight_layout()
plt.savefig("dendrogram_raw.png")
plt.close()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

linked_scaled = linkage(X_scaled, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(linked_scaled, truncate_mode='lastp', p=30, leaf_rotation=90, leaf_font_size=10)
plt.title("Hierarchical Clustering Dendrogram (Standardized Data)")
plt.tight_layout()
plt.savefig("dendrogram_standardized.png")
plt.close()
