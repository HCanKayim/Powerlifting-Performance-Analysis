import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_csv('sample_openpowerlifting.csv')

df_clean = df.dropna(subset=['TotalKg', 'Age', 'BodyweightKg'])

features = ['Age', 'BodyweightKg', 'Sex', 'Equipment', 'Event']
target = 'TotalKg'

df_features = pd.get_dummies(df_clean[features], drop_first=True)
df_target = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'KNN Regression': KNeighborsRegressor(n_neighbors=5),
    'Decision Tree Regression': DecisionTreeRegressor(random_state=42),
    'Random Forest Regression': RandomForestRegressor(random_state=42, n_estimators=100)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'RMSE': rmse, 'R2': r2}

def tune_knn_neighbors(X_train, y_train, X_test, y_test, max_neighbors=50):
    best_r2 = -np.inf
    best_n = None
    for n in range(1, max_neighbors + 1):
        knn = KNeighborsRegressor(n_neighbors=n)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        if r2 > best_r2:
            best_r2 = r2
            best_n = n
    return best_n, best_r2

best_n_neighbors, best_r2_score = tune_knn_neighbors(X_train, y_train, X_test, y_test)

knn_best = KNeighborsRegressor(n_neighbors=best_n_neighbors)
knn_best.fit(X_train, y_train)
y_pred_knn_best = knn_best.predict(X_test)
mse_knn_best = mean_squared_error(y_test, y_pred_knn_best)
rmse_knn_best = np.sqrt(mse_knn_best)
r2_knn_best = r2_score(y_test, y_pred_knn_best)

results['KNN Regression (best n_neighbors)'] = {'MSE': mse_knn_best, 'RMSE': rmse_knn_best, 'R2': r2_knn_best}

for model_name, metrics in results.items():
    print(f"{model_name}: MSE={metrics['MSE']:.2f}, RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.3f}")

print(f"\nBest n_neighbors for KNN: {best_n_neighbors} with R2={best_r2_score:.3f}")

print("\nModel Performance Comparison:")
sorted_results = sorted(results.items(), key=lambda x: x[1]['R2'], reverse=True)
for model_name, metrics in sorted_results:
    print(f"{model_name:35} | R2: {metrics['R2']:.3f} | RMSE: {metrics['RMSE']:.2f}")
