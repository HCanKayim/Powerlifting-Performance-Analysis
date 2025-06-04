import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_data(filename='sample_openpowerlifting.csv'):
    data = pd.read_csv(filename)
    
    # Select numerical features only for this analysis
    numerical_features = ['Age', 'BodyweightKg', 'WeightClassKg', 'Best3SquatKg', 
                         'Best3BenchKg', 'Best3DeadliftKg', 'TotalKg', 'Wilks']
    
    # Create feature matrix X
    X = data[numerical_features].copy()
    
    # Convert columns to numeric, replacing errors with NaN
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Create target variable y (Sex: M=0, F=1)
    le = LabelEncoder()
    y = le.fit_transform(data['Sex'])
    
    # Handle any missing values
    X = X.fillna(X.mean())
    
    return X, y, numerical_features

def perform_pca(X):
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    print("\nExplained Variance Ratio for first 5 components:")
    for i in range(5):
        print(f"PC{i+1}: {explained_variance_ratio[i]:.4f}")
    
    print(f"\nCumulative Explained Variance Ratio for first 5 components: {cumulative_variance_ratio[4]:.4f}")
    
    # Plot explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance Ratio vs Number of Components')
    plt.grid(True)
    plt.savefig('pca_explained_variance.png')
    plt.close()
    
    return X_pca, pca

def evaluate_rf_with_pca(X_pca, y):
    # Use first 5 principal components
    X_pca_5 = X_pca[:, :5]
    
    # Initialize Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Perform cross-validation
    cv_scores = cross_val_score(rf, X_pca_5, y, cv=5)
    
    print("\nRandom Forest Cross-validation results using first 5 Principal Components:")
    print(f"Individual CV scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores

def plot_feature_contributions(pca, feature_names):
    # Get the loadings (feature contributions) for first 5 PCs
    loadings = pd.DataFrame(
        pca.components_[:5].T,
        columns=[f'PC{i+1}' for i in range(5)],
        index=feature_names
    )
    
    plt.figure(figsize=(12, 8))
    plt.imshow(loadings, cmap='RdBu', aspect='auto')
    plt.colorbar()
    plt.xticks(range(5), [f'PC{i+1}' for i in range(5)])
    plt.yticks(range(len(feature_names)), feature_names)
    plt.title('Feature Contributions to First 5 Principal Components')
    
    # Add the values in the cells
    for i in range(len(feature_names)):
        for j in range(5):
            plt.text(j, i, f'{loadings.iloc[i, j]:.2f}', 
                    ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('pca_feature_contributions.png')
    plt.close()

def main():
    # Load data
    X, y, feature_names = load_data()
    print("Original dataset shape:", X.shape)
    
    # Perform PCA
    X_pca, pca = perform_pca(X)
    
    # Evaluate Random Forest with PCA components
    cv_scores = evaluate_rf_with_pca(X_pca, y)
    
    # Plot feature contributions
    plot_feature_contributions(pca, feature_names)

if __name__ == "__main__":
    main() 