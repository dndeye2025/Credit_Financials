"""
Tableau de bord - Prediction du risque de defaut de paiement
Version modernisee : design personnalise, jauge de risque, importance
des variables, vue d'ensemble des donnees, historique de session.
"""

import json

import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# -----------------------------------------------------------
# Configuration de la page
# -----------------------------------------------------------
st.set_page_config(
    page_title="Risque de credit",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------
# Style personnalise
# -----------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #ffffff; }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 0;
    }
    .subtitle {
        color: #5f6368;
        font-size: 1rem;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: linear-gradient(145deg, #f7f8fa, #eef0f3);
        border: 1px solid #e0e3e8;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        text-align: center;
    }
    .metric-card .label {
        color: #5f6368;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-card .value {
        color: #1a1a1a;
        font-size: 1.7rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }

    .verdict-card {
        border-radius: 16px;
        padding: 1.5rem 1.8rem;
        margin-top: 0.5rem;
    }
    .verdict-danger {
        background: linear-gradient(145deg, #fdecec, #fbdada);
        border: 1px solid #f3a9a9;
    }
    .verdict-ok {
        background: linear-gradient(145deg, #e9f9ef, #d9f5e3);
        border: 1px solid #a3e0bb;
    }
    .verdict-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .verdict-danger .verdict-title { color: #d92c2c; }
    .verdict-ok .verdict-title { color: #1f9d55; }
    .verdict-text { color: #3c4043; font-size: 0.95rem; }

    section[data-testid="stSidebar"] {
        background-color: #f7f8fa;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# Chargement du modele, des metriques et des donnees
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_metrics():
    with open("metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_data():
    return pd.read_csv("credit_risk_dataset.csv")


model = load_model()
metrics = load_metrics()
df = load_data()

if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------------------------------------
# Sidebar
# -----------------------------------------------------------
with st.sidebar:
    st.markdown("### 💳 Risque de Credit")
    st.caption("Projet Big Data — modele Random Forest")
    st.divider()
    st.markdown("**Performance du modele**")
    st.metric("AUC (test)", f"{metrics['test_auc']:.3f}")
    st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    st.divider()
    st.caption(
        "Pipeline : imputation → encodage → standardisation → "
        "Random Forest, optimise par validation croisee (5 folds)."
    )
    st.caption(f"{metrics['n_train']:,} clients en entrainement · "
               f"{metrics['n_test']:,} en test")

# -----------------------------------------------------------
# En-tete
# -----------------------------------------------------------
st.markdown('<p class="main-title">Tableau de bord — Risque de defaut de paiement</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Estimez le risque qu’un client fasse defaut sur son pret, '
    'a partir des donnees de son dossier.</p>',
    unsafe_allow_html=True,
)

tab_predict, tab_overview, tab_history = st.tabs(
    ["🔍 Prediction", "📊 Vue d'ensemble", "🕒 Historique"]
)

# =============================================================
# ONGLET 1 — PREDICTION
# =============================================================
with tab_predict:
    col_form, col_result = st.columns([1.1, 1])

    with col_form:
        with st.form("credit_form"):
            st.markdown("#### Informations sur le client")

            c1, c2 = st.columns(2)
            with c1:
                person_age = st.number_input("Age", 18, 100, 30)
                person_income = st.number_input("Revenu annuel ($)", 0, value=50000, step=1000)
                person_home_ownership = st.selectbox(
                    "Statut de logement", ["RENT", "OWN", "MORTGAGE", "OTHER"]
                )
                person_emp_length = st.number_input(
                    "Anciennete d'emploi (annees)", 0.0, 60.0, 5.0
                )
                cb_person_cred_hist_length = st.number_input(
                    "Anciennete du dossier de credit (annees)", 0, 40, 5
                )
                cb_person_default_on_file = st.selectbox(
                    "Defaut anterieur enregistre ?", ["N", "Y"]
                )
            with c2:
                loan_intent = st.selectbox(
                    "Motif du pret",
                    ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
                     "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
                )
                loan_grade = st.selectbox("Note du pret", ["A", "B", "C", "D", "E", "F", "G"])
                loan_amnt = st.number_input("Montant du pret ($)", 500, value=10000, step=500)
                loan_int_rate = st.number_input("Taux d'interet (%)", 0.0, 40.0, 11.0)
                loan_percent_income = st.number_input(
                    "Part du revenu consacree au pret", 0.0, 1.0, 0.2
                )

            submitted = st.form_submit_button("Analyser le dossier", use_container_width=True)

    with col_result:
        st.markdown("#### Resultat")

        if not submitted and not st.session_state.history:
            st.info("Remplissez le formulaire a gauche puis cliquez sur "
                     "**Analyser le dossier** pour voir le resultat ici.")

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

            prediction = int(model.predict(client)[0])
            proba_defaut = float(model.predict_proba(client)[0][1])

            st.session_state.last_result = (client, prediction, proba_defaut)
            st.session_state.history.insert(0, {
                "age": person_age,
                "revenu": person_income,
                "montant_pret": loan_amnt,
                "probabilite_defaut": round(proba_defaut * 100, 1),
                "verdict": "En defaut" if prediction == 1 else "En regle",
            })

        if "last_result" in st.session_state:
            client, prediction, proba_defaut = st.session_state.last_result

            # Jauge de risque
            gauge_color = "#d92c2c" if proba_defaut >= 0.5 else "#1f9d55"
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba_defaut * 100,
                number={"suffix": " %", "font": {"color": "#1a1a1a", "size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#5f6368"},
                    "bar": {"color": gauge_color},
                    "bgcolor": "#f0f2f6",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30], "color": "#e9f9ef"},
                        {"range": [30, 60], "color": "#fef6e0"},
                        {"range": [60, 100], "color": "#fdecec"},
                    ],
                },
                title={"text": "Probabilite de defaut", "font": {"color": "#5f6368", "size": 14}},
            ))
            fig.update_layout(
                height=230,
                margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#1a1a1a"},
            )
            st.plotly_chart(fig, use_container_width=True)

            if prediction == 1:
                st.markdown("""
                <div class="verdict-card verdict-danger">
                    <div class="verdict-title">🚨 ALERTE — Client en defaut de paiement</div>
                    <div class="verdict-text">Le modele estime un risque eleve que ce client
                    ne rembourse pas son pret. Une analyse manuelle supplementaire est
                    recommandee avant d'accorder ce credit.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="verdict-card verdict-ok">
                    <div class="verdict-title">✅ Client en regle</div>
                    <div class="verdict-text">Le modele estime un risque faible de defaut
                    de paiement pour ce client, sur la base des donnees fournies.</div>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("Voir le detail des donnees soumises"):
                st.dataframe(client.T.rename(columns={0: "Valeur"}), use_container_width=True)

# =============================================================
# ONGLET 2 — VUE D'ENSEMBLE
# =============================================================
with tab_overview:
    st.markdown("#### Statistiques sur les donnees d'entrainement")

    k1, k2, k3, k4 = st.columns(4)
    default_rate = (df["loan_status"] == 1).mean()
    with k1:
        st.markdown(f'<div class="metric-card"><div class="label">Clients</div>'
                     f'<div class="value">{len(df):,}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="metric-card"><div class="label">Taux de defaut</div>'
                     f'<div class="value">{default_rate:.1%}</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="metric-card"><div class="label">AUC du modele</div>'
                     f'<div class="value">{metrics["test_auc"]:.3f}</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="metric-card"><div class="label">Accuracy</div>'
                     f'<div class="value">{metrics["accuracy"]:.1%}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Importance des variables (Random Forest)**")
        imp_df = pd.DataFrame(metrics["feature_importance"])
        fig_imp = px.bar(
            imp_df.sort_values("importance"),
            x="importance", y="variable", orientation="h",
            color="importance", color_continuous_scale=["#dbe4f0", "#1f77d0"],
        )
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#1a1a1a"}, showlegend=False, coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10), height=380,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_right:
        st.markdown("**Repartition des clients selon le statut**")
        status_df = df["loan_status"].map({0: "En regle", 1: "En defaut"}).value_counts().reset_index()
        status_df.columns = ["statut", "nombre"]
        fig_pie = px.pie(
            status_df, names="statut", values="nombre", hole=0.55,
            color="statut",
            color_discrete_map={"En regle": "#1f9d55", "En defaut": "#d92c2c"},
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font={"color": "#1a1a1a"},
            margin=dict(l=10, r=10, t=10, b=10), height=380,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("**Montant du pret selon le statut du client**")
    fig_box = px.box(
        df, x=df["loan_status"].map({0: "En regle", 1: "En defaut"}), y="loan_amnt",
        color=df["loan_status"].map({0: "En regle", 1: "En defaut"}),
        color_discrete_map={"En regle": "#1f9d55", "En defaut": "#d92c2c"},
        labels={"x": "Statut", "loan_amnt": "Montant du pret ($)"},
    )
    fig_box.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1a1a1a"}, showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10), height=350,
    )
    st.plotly_chart(fig_box, use_container_width=True)

# =============================================================
# ONGLET 3 — HISTORIQUE
# =============================================================
with tab_history:
    st.markdown("#### Dossiers analyses pendant cette session")

    if not st.session_state.history:
        st.info("Aucun dossier analyse pour l'instant. Allez dans l'onglet "
                 "**Prediction** pour commencer.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        if st.button("🗑️ Effacer l'historique"):
            st.session_state.history = []
            st.rerun()
