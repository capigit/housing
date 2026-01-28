import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

st.set_page_config(page_title="California Housing - EDA", layout="wide")

st.title("Analyse Exploratoire des Données")
st.write("Dataset : California Housing")

# Chargement des données
@st.cache_data
def load_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    return df

df = load_data()

# Aperçu des données
st.subheader("Aperçu du dataset")
st.dataframe(df.head())

# Dimensions
st.subheader("Dimensions du dataset")
st.write(f"Nombre de lignes : {df.shape[0]}")
st.write(f"Nombre de colonnes : {df.shape[1]}")

# Types de variables
st.subheader("Types des variables")
st.write(df.dtypes)

# Valeurs manquantes
st.subheader("Valeurs manquantes")
st.write(df.isnull().sum())

# Statistiques descriptives
# =========================
st.subheader("Statistiques descriptives")
st.dataframe(df.describe())

# Distribution de la cible
st.subheader("Distribution de la valeur médiane des maisons")

fig, ax = plt.subplots()
ax.hist(df["MedHouseVal"], bins=50)
ax.set_xlabel("MedHouseVal")
ax.set_ylabel("Fréquence")
st.pyplot(fig)

# Corrélation
st.subheader("Matrice de corrélation")

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
st.pyplot(fig)