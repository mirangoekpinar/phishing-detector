import streamlit as st
import joblib
import os
import sys

# 🔧 Pfad zu /src/ hinzufügen (für clean_text)
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_path)

from preprocessing import clean_text  # jetzt funktioniert der Import

# 📂 Modell und Vektorizer laden
model_path = 'models/model.pkl'
vectorizer_path = 'models/vectorizer.pkl'

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(open(model_path, 'rb'))
    vectorizer = joblib.load(open(vectorizer_path, 'rb'))
else:
    st.error("❌ Modell oder Vektorisierer nicht gefunden!")
    st.stop()

# 🎯 App-Titel & Beschreibung
st.title("📧 Phishing-Mail-Detector")
st.markdown("Gib den Text einer E-Mail ein und erfahre, ob sie verdächtig ist.")

# ✍️ Texteingabe
user_input = st.text_area("E-Mail-Inhalt:")

# 🔍 Analysebutton
if st.button("Analysieren"):
    if user_input.strip() == "":
        st.warning("⚠️ Bitte gib einen E-Mail-Text ein.")
    else:
        # 🧹 Text bereinigen
        cleaned_text = clean_text(user_input)

        # 🧾 Anzeigen: Original vs. Cleaned
        st.subheader("📄 Originaltext:")
        st.write(user_input)

        st.subheader("🧹 Bereinigter Text:")
        st.write(cleaned_text)

        # 🔢 In Merkmale umwandeln
        X = vectorizer.transform([cleaned_text])

        # 🤖 Modellvorhersage
        prediction = model.predict(X)
        proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None

        # ✅ Ergebnis anzeigen
        st.subheader("📊 Ergebnis:")
        if prediction[0] == 1:
            st.error("🚨 Warnung: Diese E-Mail könnte **Phishing** sein!")
        else:
            st.success("✅ Diese E-Mail sieht **unauffällig** aus.")

        # 🔍 (Optional) Wahrscheinlichkeit anzeigen
        if proba is not None:
            phishing_proba = proba[1] * 100
            st.caption(f"🔎 Wahrscheinlichkeit für Phishing: **{phishing_proba:.2f}%**")
