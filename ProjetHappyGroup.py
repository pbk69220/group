import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Charger les données
@st.cache_data
def charger_donnees():
    data = pd.read_csv('happycommun1.csv')
    return data

data = charger_donnees()

# Ajouter une barre latérale avec des boutons de navigation
st.sidebar.header("Le bonheur dans le monde\n")
st.sidebar.markdown("Data Analyse - CDA Juin 2024<br><b>Auteurs:</b> Hélène, Patrick, Lodavé et Gaelle", unsafe_allow_html=True)
menu = st.sidebar.radio("Aller à :", ["Introduction", "DataViz", "Machine Learning", "Conclusion"])

# Section "Introduction"
if menu == "Introduction":
    st.markdown("<h1 style='font-size: 24px;'>Introduction</h1>", unsafe_allow_html=True)
    st.write("""
        Bienvenue sur cette plateforme de visualisation des données mondiales du bonheur.

Le bonheur est un concept universellement recherché, mais sa définition varie d'une personne à l'autre et d'une culture à l'autre. 
Pour les décideurs politiques, les organisations et les individus, comprendre les facteurs qui influencent le bien-être des populations est crucial.
    """)
    st.markdown("""
        <h2 style='font-size: 20px;'>
        -  Mais, comment mesurer cette notion abstraite ? <br>
        -  Peut-on quantifier le bonheur, ou reste-t-il un sentiment subjectif que l’on ne peut pleinement capturer ?
        </h2>
    """, unsafe_allow_html=True)

# Section "DataViz"
elif menu == "DataViz":
    st.markdown("<h1 style='font-size: 24px;'>Visualisations des Données</h1>", unsafe_allow_html=True)

    st.write("""
        Cette section présente des visualisations interactives pour explorer les facteurs liés au bonheur.
    """)

    indicateur = st.selectbox("Choisissez un indicateur ", data.columns[1:], index=0)

    fig = px.scatter(
        data, 
        x=indicateur, 
        y="Happiness Score", 
        title=f"Relation entre le bonheur et {indicateur}",
        labels={indicateur: indicateur, "Happiness Score": "Score de bonheur"},
        color="Region"
    )
    st.plotly_chart(fig)

# Section "Machine Learning"
elif menu == "Machine Learning":
    st.markdown("<h1 style='font-size: 24px;'>Prédiction avec Machine Learning</h1>", unsafe_allow_html=True)

    st.write("""
        Cette section utilise un modèle de régression linéaire pour prédire le score de bonheur à partir d'autres indicateurs.
    """)

    features = st.multiselect("Choisissez les variables prédictives", options=data.columns[1:], default=data.columns[1:3])

    if features:
        X = data[features]
        y = data["Happiness Score"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        st.write(f"Erreur quadratique moyenne (MSE) : {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"Coefficient de détermination (R2) : {r2_score(y_test, y_pred):.2f}")

        st.write("### Coefficients du modèle")
        coef_df = pd.DataFrame({"Variable": features, "Coefficient": model.coef_})
        st.table(coef_df)

# Section "Conclusion"
elif menu == "Conclusion":
    st.markdown("<h1 style='font-size: 24px;'>Conclusion</h1>", unsafe_allow_html=True)

    st.write("""
        Merci d'avoir exploré cette analyse du bonheur mondial avec nous. 
        Les visualisations et modèles montrent que le bonheur est influencé par divers facteurs, 
        mais il reste un sujet complexe et nuancé.
    """)
