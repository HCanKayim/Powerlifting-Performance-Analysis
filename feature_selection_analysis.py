import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load and preprocess the data
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
    
    return X, y

def analyze_mutual_info(X, y):
    # Calculate mutual information scores
    mi_scores = mutual_info_classif(X, y)
    
    # Create a dataframe of features and their MI scores
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': mi_scores
    }).drop_duplicates()
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    print("\n=== Mutual Information Feature Selection ===")
    print("All features and their Mutual Information scores:")
    print(feature_importance.to_string(index=False))
    
    return feature_importance

def analyze_rf_importance(X, y):
    # Train Random Forest and get feature importances
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Create a dataframe of features and their importance scores
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).drop_duplicates()
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    print("\n=== Random Forest Feature Selection ===")
    print("All features and their importance scores:")
    print(feature_importance.to_string(index=False))
    
    return feature_importance

def evaluate_model(X, y, selected_features, method_name):
    # Create dataset with selected features
    X_selected = X[selected_features]
    
    # Initialize Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Perform cross-validation
    cv_scores = cross_val_score(rf, X_selected, y, cv=5)
    
    print(f"\nRandom Forest Cross-validation results using top 5 {method_name} features:")
    print(f"Features used: {selected_features}")
    print(f"Individual CV scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores

def plot_comparison(mi_importance, rf_importance):
    plt.figure(figsize=(12, 6))
    
    # Plot both feature importance methods
    plt.subplot(1, 2, 1)
    plt.bar(range(5), mi_importance['importance'].head())
    plt.xticks(range(5), mi_importance['feature'].head(), rotation=45)
    plt.title('Top 5 Features (Mutual Information)')
    plt.ylabel('Mutual Information Score')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(5), rf_importance['importance'].head())
    plt.xticks(range(5), rf_importance['feature'].head(), rotation=45)
    plt.title('Top 5 Features (Random Forest)')
    plt.ylabel('Feature Importance Score')
    
    plt.tight_layout()
    plt.savefig('feature_selection_comparison.png')
    plt.close()

def main():
    # Load data
    X, y = load_data()
    print("Original dataset shape:", X.shape)
    
    # 1. Mutual Information Analysis
    mi_importance = analyze_mutual_info(X, y)
    top_5_mi = mi_importance['feature'].head().tolist()
    mi_cv_scores = evaluate_model(X, y, top_5_mi, "Mutual Information")
    
    # 2. Random Forest Feature Importance Analysis
    rf_importance = analyze_rf_importance(X, y)
    top_5_rf = rf_importance['feature'].head().tolist()
    rf_cv_scores = evaluate_model(X, y, top_5_rf, "Random Forest")
    
    # Plot comparison
    plot_comparison(mi_importance, rf_importance)

if __name__ == "__main__":
    main() 