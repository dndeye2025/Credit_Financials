"""
Tableau de bord Streamlit - Prediction du risque de defaut de paiement
"""

import streamlit as st
import pandas as pd
import joblib

# -----------------------------------------------------------
# Configuration de la page
# -----------------------------------------------------------
st.set_page_config(
    page_title="Risque de credit",
    page_icon="💳",
    layout="centered",
)

# -----------------------------------------------------------
# Chargement du modele (mis en cache pour ne pas le recharger
# a chaque interaction)
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("💳 Tableau de bord - Risque de defaut de paiement")
st.write(
    "Renseignez les informations du client pour estimer s'il presente "
    "un risque de defaut de paiement sur son pret."
)

st.divider()

# -----------------------------------------------------------
# Formulaire de saisie
# -----------------------------------------------------------
with st.form("credit_form"):
    st.subheader("Informations sur le client")

    col1, col2 = st.columns(2)

    with col1:
        person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
        person_income = st.number_input(
            "Revenu annuel ($)", min_value=0, value=50000, step=1000
        )
        person_home_ownership = st.selectbox(
            "Statut de logement",
            ["RENT", "OWN", "MORTGAGE", "OTHER"],
        )
        person_emp_length = st.number_input(
            "Anciennete d'emploi (annees)", min_value=0.0, max_value=60.0, value=5.0
        )
        cb_person_cred_hist_length = st.number_input(
            "Anciennete du dossier de credit (annees)", min_value=0, max_value=40, value=5
        )
        cb_person_default_on_file = st.selectbox(
            "Defaut de paiement anterieur enregistre ?", ["N", "Y"]
        )

    with col2:
        loan_intent = st.selectbox(
            "Motif du pret",
            ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
             "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
        )
        loan_grade = st.selectbox("Note du pret", ["A", "B", "C", "D", "E", "F", "G"])
        loan_amnt = st.number_input(
            "Montant du pret ($)", min_value=500, value=10000, step=500
        )
        loan_int_rate = st.number_input(
            "Taux d'interet (%)", min_value=0.0, max_value=40.0, value=11.0
        )
        loan_percent_income = st.number_input(
            "Part du revenu consacree au pret (0 a 1)",
            min_value=0.0, max_value=1.0, value=0.2,
        )

    submitted = st.form_submit_button("Analyser le dossier", use_container_width=True)

# -----------------------------------------------------------
# Prediction et tableau de bord
# -----------------------------------------------------------
if submitted:
    client = pd.DataFrame([{
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
    }])

    prediction = model.predict(client)[0]
    proba_defaut = model.predict_proba(client)[0][1]

    st.divider()
    st.subheader("Resultat de l'analyse")

    col_a, col_b = st.columns([2, 1])
    with col_b:
        st.metric("Probabilite de defaut", f"{proba_defaut * 100:.1f} %")

    with col_a:
        if prediction == 1:
            st.error(
                "⚠️ ALERTE : cette personne est **en defaut de paiement** "
                "(risque eleve de non-remboursement).",
                icon="🚨",
            )
        else:
            st.success(
                "✅ Cette personne est **en regle** "
                "(risque de defaut faible).",
                icon="✅",
            )

    st.progress(min(int(proba_defaut * 100), 100))

    with st.expander("Voir le detail des donnees soumises"):
        st.dataframe(client.T.rename(columns={0: "Valeur"}))
