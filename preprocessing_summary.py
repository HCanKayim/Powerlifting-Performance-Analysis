import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('sample_openpowerlifting.csv')

numeric_summary = df.describe(include=[np.number])
missing = df.isnull().sum()
missing = missing[missing > 0]
categorical_cols = df.select_dtypes(include='object').columns.tolist()
cat_missing = df[categorical_cols].isnull().sum()
cat_missing = cat_missing[cat_missing > 0]

with open("summary_report.txt", "w") as f:
    f.write("=== Summary Statistics (Numerical Columns) ===\n")
    f.write(numeric_summary.to_string())
    f.write("\n\n=== Missing Values ===\n")
    f.write(missing.to_string() if not missing.empty else "No missing values.")
    f.write("\n\n=== Categorical Columns ===\n")
    f.write(", ".join(categorical_cols))
    f.write("\n\n=== Missing in Categorical Columns ===\n")
    f.write(cat_missing.to_string() if not cat_missing.empty else "No missing values in categorical columns.")

key_numeric = ['Age', 'BodyweightKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'TotalKg']

plt.figure(figsize=(15, 10))
for i, col in enumerate(key_numeric):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df[col].dropna(), kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("distribution_plots.png")
plt.close()
