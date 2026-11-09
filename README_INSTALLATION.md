# Trading Analyzer Pro Web — V1

Application web responsive construite avec Streamlit. Elle fonctionne dans un navigateur sur ordinateur et smartphone.

## Fonctionnalités V1
- Actions, indices, Forex, crypto et matières premières via tickers Yahoo Finance
- Chandeliers, volumes, SMA 20/50, Bollinger, VWAP calculé sur la période chargée
- RSI, MACD, ATR
- Supports/résistances par pivots locaux, pivots classiques
- Profil de volume approximatif : POC, VAH, VAL
- Score de confluence indicatif et détection de quelques figures simples
- Calculateur de risque et ratio rendement/risque
- Journal de trading avec export CSV

## Limites importantes
Les données gratuites Yahoo Finance ne sont pas garanties en temps réel et peuvent être retardées, limitées ou indisponibles. Le VWAP, le profil de volume, les supports/résistances et le score sont des approximations. Il n'y a pas encore de backtesting robuste, de notifications persistantes, de base de données durable, ni de connexion à un courtier. Aucun ordre n'est exécuté.

## Lancer en local (facultatif)
Python 3.11 ou 3.12 :
```bash
python -m venv .venv
```
Windows :
```powershell
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Mac/Linux :
```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Mise en ligne gratuite avec Streamlit Community Cloud
1. Créer un compte GitHub et un dépôt **privé** ou public nommé `trading-analyzer-pro-web`.
2. Décompresser le projet et envoyer `app.py`, `requirements.txt` et le dossier `.streamlit` dans le dépôt.
3. Ouvrir https://share.streamlit.io/ et se connecter avec GitHub.
4. Choisir **Create app**, sélectionner le dépôt, la branche `main` et le fichier `app.py`.
5. Cliquer sur **Deploy**. Streamlit fournit une adresse web HTTPS utilisable sur PC et Android.
6. Sur Android/Chrome : ouvrir l'adresse, menu ⋮, puis **Ajouter à l'écran d'accueil**.

Le déploiement nécessite une intervention avec tes comptes. Ne partage jamais de mot de passe ou de clé privée dans le chat. Cette V1 n'a besoin d'aucune clé API.

## Structure
- `app.py` : application
- `requirements.txt` : dépendances
- `.streamlit/config.toml` : thème et configuration
- `.gitignore` : fichiers à ne pas publier

## Suite recommandée
Avant d'utiliser des signaux pour prendre des positions réelles : construire un moteur de backtesting tenant compte des frais, spread, slippage et biais d'anticipation, puis tester hors échantillon. Ajouter ensuite persistance du journal et notifications.
