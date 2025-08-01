import streamlit as st
import joblib

# Charger le modèle et le vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Titre de l'application
st.set_page_config(page_title="Détection de Spam", page_icon="📧")
st.title("📧 Détection de Spam par IA")
st.markdown("Saisissez un message pour analyser s'il s'agit d'un **SPAM** ou d'un **HAM** (message normal).")

# Zone de saisie du message
user_input = st.text_area("✍️ Message à analyser :", height=150)

# Bouton d'analyse
if st.button("Analyser"):
    if user_input.strip() == "":
        st.warning("⛔ Veuillez entrer un message pour l'analyser.")
    else:
        # Transformer le message avec le vectorizer
        X_input = vectorizer.transform([user_input])

        # Prédiction
        prediction = model.predict(X_input)[0]

        # Afficher la classe prédite
        if prediction == 1 or prediction == "spam":
            st.error("🚨 Ce message est un **SPAM**.")
        else:
            st.success("✅ Ce message est **HAM** (non spam).")

        # Afficher les probabilités si possible
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0]
            st.markdown("### 🔢 Probabilités de prédiction :")
            st.write(f"- HAM : {proba[0]*100:.2f}%")
            st.write(f"- SPAM : {proba[1]*100:.2f}%")
        else:
            st.info("ℹ️ Le modèle ne fournit pas de probabilité (par exemple, SVM sans calibration).")
