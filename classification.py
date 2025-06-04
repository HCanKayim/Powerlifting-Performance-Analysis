import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

# Load data
df = pd.read_csv("sample_openpowerlifting.csv")

# Drop rows with missing essential data
df_clean = df.dropna(subset=['Wilks', 'Age', 'BodyweightKg'])

# Create binary target: 1 if Wilks score above median
median_wilks = df_clean['Wilks'].median()
df_clean = df_clean.copy()
df_clean['StrongLifter'] = (df_clean['Wilks'] > median_wilks).astype(int)

# Check class balance
print("\nClass Balance Check:")
class_counts = df_clean['StrongLifter'].value_counts()
print("Class distribution before balancing:")
print(class_counts)
print(f"Class 0: {class_counts[0]} instances ({class_counts[0]/len(df_clean)*100:.1f}%)")
print(f"Class 1: {class_counts[1]} instances ({class_counts[1]/len(df_clean)*100:.1f}%)")

# Balance dataset by downsampling majority class
class_0 = df_clean[df_clean['StrongLifter'] == 0]
class_1 = df_clean[df_clean['StrongLifter'] == 1]
min_class_size = min(len(class_0), len(class_1))

# Downsample majority class
if len(class_0) > len(class_1):
    class_0 = class_0.sample(n=min_class_size, random_state=42)
else:
    class_1 = class_1.sample(n=min_class_size, random_state=42)

# Combine balanced classes
df_balanced = pd.concat([class_0, class_1])

print("\nClass distribution after balancing:")
balanced_counts = df_balanced['StrongLifter'].value_counts()
print(f"Class 0: {balanced_counts[0]} instances ({balanced_counts[0]/len(df_balanced)*100:.1f}%)")
print(f"Class 1: {balanced_counts[1]} instances ({balanced_counts[1]/len(df_balanced)*100:.1f}%)")

# Feature columns
features = ['Age', 'BodyweightKg', 'Sex', 'Equipment', 'Event']
target = 'StrongLifter'

# Encode categorical variables
df_features = pd.get_dummies(df_balanced[features], drop_first=True)
df_target = df_balanced[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.2, random_state=42)

# Define models
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN Classifier': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree Classifier': DecisionTreeClassifier(random_state=42),
    'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Evaluate on test set
classification_results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    classification_results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall
    }

# Print test set performance
print("\nClassification Results (Using Balanced Dataset)")
print(f"Median Wilks Score: {median_wilks:.2f}")
for model_name, metrics in classification_results.items():
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {metrics['Accuracy']:.3f}")
    print(f"  Precision: {metrics['Precision']:.3f}")
    print(f"  Recall:    {metrics['Recall']:.3f}")

print("\n10-Fold Cross-Validation Results")

# 10-fold cross-validation
log_reg = LogisticRegression(max_iter=1000, random_state=42)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

log_scores = cross_val_score(log_reg, df_features, df_target, cv=10, scoring='accuracy')
rf_scores = cross_val_score(rf_clf, df_features, df_target, cv=10, scoring='accuracy')

# Print CV results
print("\nLogistic Regression:")
print("Fold Accuracies:", [f"{score:.3f}" for score in log_scores])
print(f"Average Accuracy: {np.mean(log_scores):.3f} (±{np.std(log_scores):.3f})")

print("\nRandom Forest:")
print("Fold Accuracies:", [f"{score:.3f}" for score in rf_scores])
print(f"Average Accuracy: {np.mean(rf_scores):.3f} (±{np.std(rf_scores):.3f})")

# Overfitting Analysis
print("\nOverfitting Analysis:")
for name, clf in {'Logistic Regression': log_reg, 'Random Forest': rf_clf}.items():
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f"\n{name}:")
    print(f"  Training Accuracy: {train_score:.3f}")
    print(f"  Test Accuracy:     {test_score:.3f}")
    print(f"  Difference:        {train_score - test_score:.3f}")

# Additional analysis of Wilks score distribution
print("\nWilks Score Distribution by Sex:")
print(df_clean.groupby('Sex')['Wilks'].describe())
