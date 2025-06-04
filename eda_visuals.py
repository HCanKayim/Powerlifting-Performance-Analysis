import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('sample_openpowerlifting.csv')

plt.figure(figsize=(10, 8))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(data=df, x='Sex', y='TotalKg')
plt.title("Total Weight Lifted by Sex")
plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='Equipment', y='TotalKg')
plt.title("Total Weight Lifted by Equipment")
plt.tight_layout()
plt.savefig("boxplots_totalkg.png")
plt.close()

top_federations = df['Federation'].value_counts().nlargest(10)
top_countries = df['Country'].value_counts().nlargest(10)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.barplot(x=top_federations.values, y=top_federations.index)
plt.title("Top 10 Federations")
plt.xlabel("Count")
plt.subplot(1, 2, 2)
sns.barplot(x=top_countries.values, y=top_countries.index)
plt.title("Top 10 Countries")
plt.xlabel("Count")
plt.tight_layout()
plt.savefig("barplots_federation_country.png")
plt.close()
