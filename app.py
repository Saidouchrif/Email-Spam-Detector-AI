import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Détection de Spam", page_icon="📧")
st.title("📧 Détection de Spam par IA")
st.markdown("Saisissez un message pour analyser s'il s'agit d'un **SPAM** ou d'un **HAM** (message normal).")

# Charger le modèle et le vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Zone de saisie du message
user_input = st.text_area("✍️ Message à analyser :", height=150)

# Bouton d'analyse
if st.button("Analyser"):
    if user_input.strip() == "":
        st.warning("⛔ Veuillez entrer un message pour l'analyser.")
    else:
        # WordCloud du message utilisateur
        st.markdown("### ☁️ Nuage de mots du message")
        wordcloud = WordCloud(width=600, height=300, background_color='white').generate(user_input)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis("off")
        st.pyplot(fig_wc)

        # Transformation et prédiction
        X_input = vectorizer.transform([user_input])
        prediction = model.predict(X_input)[0]

        # Afficher la classe prédite
        if prediction == 1 or prediction == "spam":
            st.error("🚨 Ce message est un **SPAM**.")
        else:
            st.success("✅ Ce message est **HAM** (non spam).")

        # Probabilités de prédiction
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0]
            labels = ["HAM", "SPAM"]
            st.markdown("### 📊 Probabilités de prédiction")
            
            # Affichage en texte
            st.write(f"- HAM : {proba[0]*100:.2f}%")
            st.write(f"- SPAM : {proba[1]*100:.2f}%")

            # Affichage en graphique
            fig, ax = plt.subplots()
            sns.barplot(x=labels, y=proba, palette="viridis", ax=ax)
            ax.set_ylabel("Probabilité")
            ax.set_ylim(0, 1)
            st.pyplot(fig)
        else:
            st.info("ℹ️ Le modèle ne fournit pas de probabilités (ex: SVM sans calibration).")
