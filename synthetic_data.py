import pandas as pd
import numpy as np

df = pd.read_csv("data/final_dataset.csv")

# Normalize bands
bands = ['B2','B3','B4','B5','B6','B7','B8','B8A']
for b in bands:
    df[b] = df[b] / 10000

# ---------------------------
# NEW HIGH-VARIANCE LABELS
# ---------------------------

# Chlorophyll — nonlinear red-edge interaction
df['Chl'] = (
    80*(df['B5']/df['B4']) +
    40*(df['B6']) +
    15*np.sin(df['B7']*10) +
    np.random.normal(0, 5, len(df))
)

# Dissolved Oxygen — depends on turbidity + algae
df['DO'] = (
    10
    - 2.5*df['B8']
    - 0.2*df['Chl']
    + 3*df['B3']
    + np.random.normal(0, 1.2, len(df))
)

# NH3 — strong variability added
df['NH3'] = (
    3*(df['B2']/df['B3']) +
    2*(df['B4']) +
    4*np.random.rand(len(df))
)

df.to_csv("data/final_dataset.csv", index=False)

print("Dataset regenerated with high variance labels")
