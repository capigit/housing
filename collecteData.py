import pandas as pd
from sklearn.datasets import fetch_california_housing

# Chargement du dataset
housing = fetch_california_housing(as_frame=True)

# Création du DataFrame
df = housing.frame

df.to_csv("data/dataset.csv", index=False)