# Trading Analyzer Pro V3.1

## Ajouts
- Trois profils : scalping (5/15 min), day trading (5/15 min/1 h), swing (4 h/jour/semaine). Les unités restent librement sélectionnables.
- Couche de données séparée : EODHD REST ou Yahoo Finance.
- Mapping EODHD explicite pour quelques actions US/Paris, paires Forex et cryptos. Les actifs non mappés sont laissés indisponibles plutôt que convertis au hasard.
- Interface courtier abstraite, sans connexion ni passage d'ordre.

## Fournisseur recommandé
EODHD Active Trader / EOD + Intraday est affiché à 29,99 USD/mois (299,90 USD/an) et inclut les données intraday ainsi qu'un WebSocket jusqu'à 50 tickers selon la page tarifaire. Vérifier le coût final en euros et les droits de chaque place : https://eodhd.com/pricing

Twelve Data Basic est gratuit mais limité ; les flux WebSocket complets sont réservés aux offres Pro et supérieures. https://twelvedata.com/pricing

## Installation Windows
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Clé API EODHD
Dans Streamlit Community Cloud : Manage app → Settings → Secrets, ajouter :
```toml
EODHD_API_KEY = "VOTRE_CLE_API"
```
En local, définir la variable d'environnement `EODHD_API_KEY`. Ne jamais publier la clé dans GitHub.

## Limites et prudence
- Ce build n'active pas encore le WebSocket EODHD. Le scalping en réel exige un flux streaming, un contrôle de latence et les spreads du courtier : le REST intraday ne suffit pas.
- EODHD REST : 1m/5m/1h intraday ; 15m/4h sont agrégés localement. Les unités supérieures sont calculées depuis les données journalières.
- Le mapping des symboles est volontairement limité ; indices et contrats futures demandent validation du ticker exact et des droits. Le scanner n'est pas exhaustif.
- Les scores sont des heuristiques, pas des probabilités de gain. Les backtests restent pédagogiques, sans validation walk-forward complète.
- Aucun compte courtier n'est branché. Pour cela, le nom exact du courtier est indispensable ; le connecteur doit commencer en lecture seule.
