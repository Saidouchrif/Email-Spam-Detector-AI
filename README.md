# 📧 Détection de Spam avec Machine Learning

Ce projet a pour objectif de développer un système de **détection automatique de spam** à partir d’emails ou de messages texte. Le modèle s'appuie sur des techniques de **traitement du langage naturel (NLP)** et des **algorithmes de classification supervisée**.

Le code est contenu dans le notebook Jupyter `Spam1.ipynb`.

---

## 🧠 Objectif

- Charger un jeu de données de messages.
- Nettoyer et prétraiter les textes (suppression des stopwords, tokenisation, stemming).
- Convertir les textes en vecteurs numériques (Bag of Words ou TF-IDF).
- Appliquer des algorithmes de classification (SVM, Random Forest, etc.).
- Rééquilibrer les classes avec SMOTE si nécessaire.
- Évaluer les performances (accuracy, précision, rappel, F1-score, matrice de confusion).
- Visualiser les résultats.

---

## 🧪 Technologies utilisées

| Type | Outils / Librairies |
|------|----------------------|
| 💻 Langage | Python 3 |
| 📊 Traitement de données | pandas, numpy |
| 📈 Visualisation | matplotlib, seaborn |
| 📚 NLP | nltk |
| ⚙️ Machine Learning | scikit-learn |
| ⚖️ Rééquilibrage des données | imbalanced-learn (SMOTE) |
| 🐳 Conteneurisation | Docker |
| 📓 IDE | Jupyter Notebook |

---

## 🧰 Installation via Docker

### 🔧 Prérequis

- [Docker](https://docs.docker.com/get-docker/) installé sur votre machine.

### 📁 Structure du projet

spam-detector/
├── app.py
├── DataSet_Emails.csv
└── README.md
└── Spam1.ipynb
└── tfidf_vectorizer.pkl