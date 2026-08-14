# Tableau de bord - Risque de credit

Application Streamlit qui predit si un client est en defaut de paiement,
a partir d'un modele Random Forest entraine sur `credit_risk_dataset.csv`.

## Contenu du projet

- `credit_risk_dataset.csv` : les donnees
- `train_model.py` : script qui entraine le modele et genere `model.pkl`
- `app.py` : l'application Streamlit (le tableau de bord)
- `requirements.txt` : les librairies necessaires
- `model.pkl` : le modele entraine (genere par train_model.py)

## 1. Tester en local

```bash
# Installer les librairies
pip install -r requirements.txt

# Entrainer le modele (genere model.pkl)
python train_model.py

# Lancer le tableau de bord
streamlit run app.py
```

Une page va s'ouvrir automatiquement dans votre navigateur (en general
http://localhost:8501).

## 2. Mettre le projet sur GitHub

Dans le dossier du projet :

```bash
git init
git add credit_risk_dataset.csv train_model.py app.py requirements.txt model.pkl README.md
git commit -m "Modele et tableau de bord de risque de credit"
git branch -M main
git remote add origin https://github.com/VOTRE-NOM-UTILISATEUR/VOTRE-REPO.git
git push -u origin main
```

(Remplacez l'URL par celle du repo que vous avez deja cree.)

Important : verifiez que `model.pkl` est bien pousse sur GitHub (il ne doit
pas depasser 100 Mo, ce qui ne sera pas le cas ici).

## 3. Deployer sur Streamlit Community Cloud (gratuit)

1. Aller sur https://share.streamlit.io
2. Se connecter avec votre compte GitHub
3. Cliquer sur "New app"
4. Choisir votre repository, la branche `main`, et le fichier principal
   `app.py`
5. Cliquer sur "Deploy"

Streamlit installe automatiquement les librairies listees dans
`requirements.txt`, puis votre application est accessible via une URL
publique du type `https://votre-app.streamlit.app`.

## Note sur le passage de R vers Python

Le travail original a ete fait en R avec sparklyr (imputation, encodage,
standardisation, Random Forest, validation croisee). Streamlit etant un
framework Python, le meme pipeline a ete reproduit avec scikit-learn
(SimpleImputer, OneHotEncoder, StandardScaler, RandomForestClassifier,
GridSearchCV) pour pouvoir etre integre a l'application.
