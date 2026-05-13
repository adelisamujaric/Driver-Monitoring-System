import pandas as pd

df = pd.read_csv("/Users/adelisa/FIT/ML/outputs/features.csv")

outlieri = df[(df['ear'] >= 1.0) | (df['mar'] >= 5.0)]
print(f"Nađeno outliera: {len(outlieri)}")
print(outlieri)

