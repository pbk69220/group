import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Charger les données
@st.cache_data
def load_data():
    try:
        # Remplacez 'happycommun1.csv' par le chemin absolu ou assurez-vous qu'il est dans le même répertoire
        data = pd.read_csv("happycommun1.csv")
        return data
    except FileNotFoundError:
        st.error("Le fichier 'happycommun1.csv' est introuvable. Assurez-vous qu'il est dans le bon répertoire.")
        return None

data = load_data()

# Menu Streamlit
menu = ["Introduction", "Visualisation", "Modélisation"]
choice = st.sidebar.selectbox("Menu", menu)

if data is not None:  # Vérifiez que les données ont été chargées correctement
    if choice == "Introduction":
        st.title("Analyse du Bonheur et du Bien-être")
        st.write("Cette application explore les facteurs influençant le bonheur à travers le monde.")
        st.write("Veuillez choisir une option dans le menu à gauche.")

    elif choice == "Visualisation":
        st.title("Visualisation des Données")

        # Afficher les données
        st.write("### Aperçu des données")
        st.dataframe(data.head())

        # Carte du monde
        st.write("### Carte du Bonheur par Pays")
        if "Country" in data.columns and "Life Ladder" in data.columns:
            fig = px.choropleth(data, locations="Country", locationmode="country names",
                                color="Life Ladder", title="Indice de Bonheur par Pays",
                                color_continuous_scale="Viridis")
            st.plotly_chart(fig)
        else:
            st.warning("Les colonnes nécessaires pour la carte (Country, Life Ladder) sont absentes.")

        # Heatmap des corrélations
        st.write("### Heatmap des Corrélations")
        corr_matrix = data.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    elif choice == "Modélisation":
        st.title("Modélisation et Prédictions")
        st.write("Sélectionnez les variables pour modéliser les données.")

        # Filtrer les colonnes pour la modélisation
        target = "Life Ladder"
        features = ["Log GDP per capita", "Social support", "Healthy life expectancy at birth", 
                    "Freedom to make life choices"]
        if all(col in data.columns for col in features + [target]):
            df = data.dropna(subset=features + [target])

            # Séparation des données
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Normalisation des données
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Modèle de régression linéaire
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # Résultats
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"### Erreur Quadratique Moyenne (MSE) : {mse:.2f}")

            # Graphique des prédictions
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.6)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
            ax.set_xlabel("Valeurs Réelles")
            ax.set_ylabel("Prédictions")
            ax.set_title("Prédictions vs Réelles")
            st.pyplot(fig)
        else:
            st.warning("Les colonnes nécessaires pour la modélisation sont absentes.")
else:
    st.stop()  # Arrête l'exécution si les données sont absentes
