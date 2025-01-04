import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
@st.cache
def load_data():
    return pd.read_csv("happycommun1.csv")
data = load_data()


# Menu Streamlit
menu = ["Introduction", "Hypothèses", "Sources", "Visualisation", "Modélisation et Prédictions", "Conclusion"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Introduction":
    st.title("Étude du Bonheur et du Bien-être Subjectif")
    st.write("Le bien-être subjectif et le bonheur sont des indicateurs clés pour évaluer la qualité de vie.")
    st.subheader("Problématique")
    st.write("Cette étude explore les facteurs influençant le bonheur mondial entre 2005 et 2023.")

elif choice == "Hypothèses":
    st.title("Hypothèses")
    st.write("### Hypothèses générales")
    st.markdown("- Les variables économiques comme le PIB par habitant influencent le bonheur.")
    st.markdown("- Les variables sociales et la liberté influencent positivement le bonheur.")
    st.write("### Hypothèses spécifiques (Europe)")
    st.markdown("- Le soutien social a un impact prononcé en Europe.")
    st.markdown("- La liberté économique et personnelle sont corrélées au bonheur.")

elif choice == "Sources":
    st.title("Sources et Données")
    st.write("Extrait des données World Happiness Report :")
    st.write(data.head())
    st.write("### Analyse des valeurs manquantes")
    st.bar_chart(data.isna().sum())

elif choice == "Visualisation":
    st.title("Visualisation des Données")

    st.write("### Carte mondiale du bonheur")
    if "Country" in data.columns and "Life Ladder" in data.columns:
        fig = px.choropleth(data, locations="Country", locationmode="country names", color="Life Ladder", title="Indice de Bonheur par Pays")
        st.plotly_chart(fig)

    st.write("### Heatmap des corrélations")
    corr_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.write("### Boxplot des Variables Sélectionnées")
    selected_vars = ["Economic Freedom", "Log GDP per capita", "Social support", "Healthy life expectancy at birth"]
    if all(var in data.columns for var in selected_vars):
        fig, ax = plt.subplots()
        sns.boxplot(data=data[selected_vars])
        st.pyplot(fig)

    st.write("### Histogramme Comparatif entre Continents")
    if "Region" in data.columns and "Life Ladder" in data.columns:
        fig = px.histogram(data, x="Region", y="Life Ladder", color="Region", barmode="group", title="Comparaison du Bonheur par Continent")
        st.plotly_chart(fig)

elif choice == "Modélisation et Prédictions":
    st.title("Modélisation et Prédictions")

    # Préparation des données
    selected_vars = ["Log GDP per capita", "Social support", "Healthy life expectancy at birth", "Freedom to make life choices"]
    target = "Life Ladder"
    df = data.dropna(subset=selected_vars + [target])

    X = df[selected_vars]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modèles
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor()
    }

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, predictions)
        st.write(f"### {name}")
        st.write(f"MSE: {mse:.2f}")

elif choice == "Conclusion":
    st.title("Conclusion")
    st.write("L'étude montre que des facteurs économiques, sociaux et politiques influencent significativement le bonheur.")
