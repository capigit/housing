import streamlit as st
import pandas as pd
from model.model import train_model, load_model
import os

st.set_page_config(page_title="Plateforme Data Science", layout="wide")

st.title("Plateforme d’Analyse et de Prédiction de Données")
st.write("Dataset : California Housing")

DATA_PATH = "data/dataset.csv"
MODEL_PATH = "model/trained_model.pkl"

# Chargement des données
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# Aperçu des données
st.subheader("Aperçu du dataset")
st.dataframe(df.head())

# Entraînement du modèle
st.subheader("Entraînement du modèle")

model_type = st.selectbox(
    "Choisir le modèle",
    ("linear", "random_forest")
)

if st.button("Entraîner le modèle"):
    rmse, r2 = train_model(model_type)
    st.success("Modèle entraîné et sauvegardé")
    st.write(f"RMSE : {rmse:.3f}")
    st.write(f"R² : {r2:.3f}")

# Prédiction
st.subheader("Prédiction")

if not os.path.exists(MODEL_PATH):
    st.warning("Veuillez entraîner le modèle avant de faire une prédiction.")
else:
    model = load_model()

    input_data = {}
    for col in df.drop("MedHouseVal", axis=1).columns:
        input_data[col] = st.number_input(col, float(df[col].mean()))

    input_df = pd.DataFrame([input_data])

    if st.button("Prédire"):
        prediction = model.predict(input_df)[0]
        st.success(f"Valeur prédite du logement : {prediction:.3f}")