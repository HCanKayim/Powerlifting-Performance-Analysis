import pandas as pd
import random
import math

df = pd.read_csv('openpowerlifting.csv')
required_columns = ['Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'Age', 'Country', 'MeetState', 'Tested']
df_clean = df.dropna(subset=required_columns)
columns_to_drop = [
    'Squat1Kg', 'Squat2Kg', 'Squat3Kg', 'Squat4Kg',
    'Bench1Kg', 'Bench2Kg', 'Bench3Kg', 'Bench4Kg',
    'Deadlift1Kg', 'Deadlift2Kg', 'Deadlift3Kg', 'Deadlift4Kg'
]
df_clean.drop(columns=columns_to_drop, inplace=True, errors='ignore')
random_state = math.floor(random.random() * 10000)
sample_df = df_clean.sample(n=2000, random_state=random_state)
sample_df.to_csv('sample_openpowerlifting.csv', index=False)
