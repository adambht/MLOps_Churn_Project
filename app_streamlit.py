import streamlit as st
import requests
import json

# FastAPI endpoint
API_URL = "http://localhost:8000/predict"

# Design premium
st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #ffffff;
            text-align: center;
        }
        .subtitle {
            font-size: 18px;
            color: #34495e;
            text-align: center;
            margin-bottom: 20px;
        }
        .stTextArea {
            border-radius: 10px;
            border: 1px solid #bdc3c7;
            padding: 10px;
            font-size: 16px;
        }
        .stButton > button {
            border-radius: 10px;
            background-color: #2980b9;
            color: white;
            font-size: 18px;
            padding: 10px;
            width: 100%;
            border: none;
        }
        .stButton > button:hover {
            background-color: #1f6690;
        }
    </style>
""", unsafe_allow_html=True)

# UI
st.markdown('<p class="title">📊 Customer Churn Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Prédisez si un client va rester ou partir 📈</p>', unsafe_allow_html=True)

# Zone de texte pour entrer la liste des features
features_input = st.text_area("🔢 Entrez les caractéristiques (séparées par des virgules)", 
                              "4.0, 3.0, 1.0, 6.0, 2.0, 8.0, 2.0, 3.0, 2.0, 6.0, 3.0, 1.0, 2.0, 1.0, 1.0, 6.0, 3.0")

# Bouton de prédiction
if st.button("✨ Prédire"):
    try:
        # Convertir l'entrée en liste de nombres
        features = [float(x.strip()) for x in features_input.split(",")]

        # Vérifier le nombre de features
        if len(features) != 17:
            st.error(f"⚠️ Vous devez entrer exactement 17 valeurs. Vous en avez entré {len(features)}.")
        else:
            # Envoie la requête à FastAPI
            payload = json.dumps({"features": features})
            headers = {"Content-Type": "application/json"}
            response = requests.post(API_URL, data=payload, headers=headers)

            if response.status_code == 200:
                prediction = response.json().get("prediction", "No prediction returned")
                st.success(f"🎯 Résultat : **{prediction}**")
            else:
                st.error(f"🚨 Erreur dans la prédiction : {response.text}")

    except ValueError:
        st.error("❌ Veuillez entrer uniquement des nombres séparés par des virgules.")
