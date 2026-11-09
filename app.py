import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone

st.set_page_config(page_title="Trading Analyzer Pro", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px;}
[data-testid="stMetric"] {background: rgba(120,120,120,.07); padding: 12px; border-radius: 10px;}
.small-note {font-size: .86rem; opacity: .78;}
</style>
""", unsafe_allow_html=True)

PRESETS = {
    "Actions": {"Apple": "AAPL", "Nvidia": "NVDA", "Microsoft": "MSFT", "TotalEnergies": "TTE.PA", "Airbus": "AIR.PA"},
    "Indices": {"Nasdaq 100": "^NDX", "S&P 500": "^GSPC", "CAC 40": "^FCHI", "DAX": "^GDAXI", "Dow Jones": "^DJI"},
    "Forex": {"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X", "AUD/USD": "AUDUSD=X"},
    "Crypto": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD"},
    "Matières premières": {"Or": "GC=F", "Argent": "SI=F", "Pétrole WTI": "CL=F", "Gaz naturel": "NG=F"},
}

PERIODS = {
    "1 jour / 5 min": ("1d", "5m"),
    "5 jours / 15 min": ("5d", "15m"),
    "1 mois / 1 heure": ("1mo", "1h"),
    "3 mois / 1 jour": ("3mo", "1d"),
    "6 mois / 1 jour": ("6mo", "1d"),
    "1 an / 1 jour": ("1y", "1d"),
    "2 ans / 1 jour": ("2y", "1d"),
    "5 ans / 1 semaine": ("5y", "1wk"),
}

@st.cache_data(ttl=300, show_spinner=False)
def load_market_data(ticker, period, interval):
    raw = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
    needed = ["open", "high", "low", "close", "volume"]
    if not all(c in raw.columns for c in needed):
        return pd.DataFrame()
    df = raw[needed].copy().dropna(subset=["open", "high", "low", "close"])
    df.index = pd.to_datetime(df.index)
    return df

def calculate_indicators(df):
    x = df.copy()
    close = x["close"]
    x["sma20"] = close.rolling(20).mean()
    x["sma50"] = close.rolling(50).mean()
    x["ema12"] = close.ewm(span=12, adjust=False).mean()
    x["ema26"] = close.ewm(span=26, adjust=False).mean()
    x["macd"] = x["ema12"] - x["ema26"]
    x["macd_signal"] = x["macd"].ewm(span=9, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi"] = 100 - (100 / (1 + rs))
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    x["bb_mid"] = mid
    x["bb_upper"] = mid + 2 * std
    x["bb_lower"] = mid - 2 * std
    prev_close = close.shift(1)
    tr = pd.concat([(x["high"]-x["low"]), (x["high"]-prev_close).abs(), (x["low"]-prev_close).abs()], axis=1).max(axis=1)
    x["atr"] = tr.rolling(14).mean()
    typical = (x["high"] + x["low"] + close) / 3
    vol = x["volume"].fillna(0)
    cumvol = vol.cumsum().replace(0, np.nan)
    x["vwap"] = (typical * vol).cumsum() / cumvol
    return x

def pivot_levels(df):
    last = df.iloc[-1]
    p = (last["high"] + last["low"] + last["close"]) / 3
    return {"Pivot": p, "R1": 2*p-last["low"], "S1": 2*p-last["high"],
            "R2": p+(last["high"]-last["low"]), "S2": p-(last["high"]-last["low"])}

def swing_levels(df, window=3):
    highs, lows = [], []
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    for i in range(window, len(df)-window):
        if h[i] == np.max(h[i-window:i+window+1]):
            highs.append(float(h[i]))
        if l[i] == np.min(l[i-window:i+window+1]):
            lows.append(float(l[i]))
    # Keep nearby distinct levels only
    def unique_sorted(vals):
        vals = sorted(vals)
        out = []
        for v in vals:
            if not out or abs(v-out[-1])/max(abs(v), 1e-9) > 0.003:
                out.append(v)
        return out
    return unique_sorted(lows)[-5:], unique_sorted(highs)[:][-5:]

def volume_profile(df, bins=30):
    typical = (df["high"] + df["low"] + df["close"]) / 3
    low, high = float(df["low"].min()), float(df["high"].max())
    if not np.isfinite(low+high) or high <= low:
        return None
    edges = np.linspace(low, high, bins+1)
    idx = np.clip(np.digitize(typical, edges)-1, 0, bins-1)
    volumes = np.zeros(bins)
    for i, v in zip(idx, df["volume"].fillna(0).to_numpy()):
        volumes[i] += max(float(v), 0)
    centers = (edges[:-1]+edges[1:])/2
    if volumes.sum() <= 0:
        return None
    poc_i = int(np.argmax(volumes))
    # Approximate 70% value area by expanding from POC to larger adjacent volume
    selected = {poc_i}
    target = volumes.sum() * 0.70
    total = volumes[poc_i]
    left, right = poc_i-1, poc_i+1
    while total < target and (left >= 0 or right < bins):
        lv = volumes[left] if left >= 0 else -1
        rv = volumes[right] if right < bins else -1
        if rv > lv:
            selected.add(right); total += max(rv, 0); right += 1
        else:
            selected.add(left); total += max(lv, 0); left -= 1
    return {"poc": float(centers[poc_i]), "vah": float(centers[max(selected)]),
            "val": float(centers[min(selected)]), "centers": centers, "volumes": volumes}

def dow_structure(df):
    lows, highs = swing_levels(df, window=3)
    # Compare the last two confirmed pivot highs and lows where available.
    # This is a simplified structural reading, not a full Dow Theory engine.
    if len(highs) >= 2 and len(lows) >= 2:
        higher_high = highs[-1] > highs[-2]
        higher_low = lows[-1] > lows[-2]
        lower_high = highs[-1] < highs[-2]
        lower_low = lows[-1] < lows[-2]
        if higher_high and higher_low:
            return "Structure haussière (sommets et creux ascendants)"
        if lower_high and lower_low:
            return "Structure baissière (sommets et creux descendants)"
        return "Structure mixte / consolidation"
    return "Données insuffisantes pour qualifier la structure de Dow"

def candle_patterns(df):
    if len(df) < 2:
        return []
    a, b = df.iloc[-2], df.iloc[-1]
    body = abs(b.close-b.open)
    rng = max(b.high-b.low, 1e-12)
    lower = min(b.open,b.close)-b.low
    upper = b.high-max(b.open,b.close)
    found = []
    if body/rng < 0.10:
        found.append("Doji")
    if lower > body*2 and upper < body*0.8 and body/rng < 0.4:
        found.append("Marteau potentiel")
    if b.close > b.open and a.close < a.open and b.open <= a.close and b.close >= a.open:
        found.append("Avalement haussier")
    if b.close < b.open and a.close > a.open and b.open >= a.close and b.close <= a.open:
        found.append("Avalement baissier")
    return found

def signal_analysis(df):
    last = df.iloc[-1]
    score = 0
    reasons = []
    if pd.notna(last.sma20) and pd.notna(last.sma50):
        if last.sma20 > last.sma50:
            score += 1; reasons.append("SMA 20 au-dessus de SMA 50")
        else:
            score -= 1; reasons.append("SMA 20 sous SMA 50")
    if pd.notna(last.vwap):
        if last.close > last.vwap:
            score += 1; reasons.append("Cours au-dessus du VWAP calculé")
        else:
            score -= 1; reasons.append("Cours sous le VWAP calculé")
    if pd.notna(last.macd) and pd.notna(last.macd_signal):
        if last.macd > last.macd_signal:
            score += 1; reasons.append("MACD au-dessus de son signal")
        else:
            score -= 1; reasons.append("MACD sous son signal")
    if pd.notna(last.rsi):
        if 50 <= last.rsi <= 70:
            score += 1; reasons.append("RSI en zone de momentum haussier")
        elif 30 <= last.rsi < 50:
            score -= 1; reasons.append("RSI en zone de momentum faible")
        elif last.rsi > 70:
            reasons.append("RSI élevé : risque de surachat")
        elif last.rsi < 30:
            reasons.append("RSI bas : possible survente")
    patterns = candle_patterns(df)
    if "Avalement haussier" in patterns: score += 1
    if "Avalement baissier" in patterns: score -= 1
    label = "Biais haussier" if score >= 2 else "Biais baissier" if score <= -2 else "Neutre / mitigé"
    return score, label, reasons, patterns

st.title("📈 Trading Analyzer Pro")
st.caption("Plateforme d'analyse technique pédagogique — données gratuites Yahoo Finance")
with st.sidebar:
    st.header("Marché")
    market = st.selectbox("Catégorie", list(PRESETS.keys()))
    preset_name = st.selectbox("Instrument", list(PRESETS[market].keys()))
    default_ticker = PRESETS[market][preset_name]
    ticker = st.text_input("Ticker Yahoo Finance", value=default_ticker).strip().upper()
    timeframe = st.selectbox("Période / unité", list(PERIODS.keys()), index=3)
    period, interval = PERIODS[timeframe]
    st.divider()
    st.caption("Exemples de tickers : AAPL, AIR.PA, EURUSD=X, BTC-USD, GC=F.")
    refresh = st.button("↻ Actualiser les données", use_container_width=True)
    if refresh:
        st.cache_data.clear()

if not ticker:
    st.error("Saisis un ticker valide.")
    st.stop()

with st.spinner(f"Chargement de {ticker} depuis Yahoo Finance…"):
    raw = load_market_data(ticker, period, interval)

if raw.empty or len(raw) < 5:
    st.error("Aucune donnée exploitable reçue. Vérifie le ticker, la période ou réessaie plus tard.")
    st.info("Les données gratuites peuvent être retardées, limitées ou indisponibles selon l'instrument.")
    st.stop()

df = calculate_indicators(raw)
last = df.iloc[-1]
prev = df.iloc[-2]
change = (last.close / prev.close - 1) * 100 if prev.close else 0
score, bias, reasons, patterns = signal_analysis(df)
profile = volume_profile(df)
supports, resistances = swing_levels(df)
dow_reading = dow_structure(df)
pivots = pivot_levels(df)

m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("Dernier cours", f"{last.close:,.4f}", f"{change:+.2f}%")
m2.metric("RSI (14)", f"{last.rsi:.1f}" if pd.notna(last.rsi) else "—")
m3.metric("ATR (14)", f"{last.atr:.4f}" if pd.notna(last.atr) else "—")
m4.metric("Biais indicatif", bias)
m5.metric("Confluence", f"{score:+d}/5")

tab_chart, tab_analysis, tab_risk, tab_journal, tab_method = st.tabs(
    ["📊 Graphique", "🧭 Analyse", "🛡️ Gestion du risque", "📓 Journal", "ℹ️ Méthode & limites"]
)

with tab_chart:
    overlays = st.multiselect("Afficher sur le graphique", ["SMA 20", "SMA 50", "Bandes de Bollinger", "VWAP"], default=["SMA 20","SMA 50","Bandes de Bollinger"])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.75,0.25], specs=[[{"secondary_y":False}],[{"secondary_y":False}]])
    fig.add_trace(go.Candlestick(x=df.index, open=df.open, high=df.high, low=df.low, close=df.close, name="Prix"), row=1,col=1)
    if "SMA 20" in overlays: fig.add_trace(go.Scatter(x=df.index,y=df.sma20,name="SMA 20",line=dict(width=1.4)),row=1,col=1)
    if "SMA 50" in overlays: fig.add_trace(go.Scatter(x=df.index,y=df.sma50,name="SMA 50",line=dict(width=1.4)),row=1,col=1)
    if "Bandes de Bollinger" in overlays:
        fig.add_trace(go.Scatter(x=df.index,y=df.bb_upper,name="Bollinger haute",line=dict(width=1,dash="dot")),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df.bb_lower,name="Bollinger basse",line=dict(width=1,dash="dot"),fill="tonexty",fillcolor="rgba(100,130,180,.08)"),row=1,col=1)
    if "VWAP" in overlays: fig.add_trace(go.Scatter(x=df.index,y=df.vwap,name="VWAP période chargée",line=dict(width=1.5,dash="dash")),row=1,col=1)
    fig.add_trace(go.Bar(x=df.index,y=df.volume,name="Volume",marker=dict(opacity=.55)),row=2,col=1)
    fig.update_layout(height=680, template="plotly_dark", xaxis_rangeslider_visible=False,
                      margin=dict(l=10,r=10,t=35,b=10), legend=dict(orientation="h"))
    fig.update_yaxes(title_text="Prix",row=1,col=1)
    fig.update_yaxes(title_text="Volume",row=2,col=1)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{len(df)} bougies chargées • Dernière bougie : {df.index[-1]} • Unité : {interval}")

with tab_analysis:
    left,right = st.columns([1.1,1])
    with left:
        st.subheader("Confluence des indicateurs")
        st.metric("Lecture indicative", bias, f"Score {score:+d}")
        for reason in reasons: st.write("•", reason)
        if patterns:
            st.write("**Figures détectées sur la dernière bougie :** " + ", ".join(patterns))
        else:
            st.write("Aucune figure simple détectée sur la dernière bougie.")
        st.warning("Ce score est une heuristique, pas une probabilité de gain ni un conseil d'achat/vente.")
    with right:
        st.subheader("Théorie de Dow — lecture simplifiée")
        st.write(dow_reading)
        st.subheader("Supports et résistances")
        st.write("**Supports détectés**")
        st.write(" · ".join(f"{v:,.4f}" for v in supports) if supports else "Pas assez de points de retournement.")
        st.write("**Résistances détectées**")
        st.write(" · ".join(f"{v:,.4f}" for v in resistances) if resistances else "Pas assez de points de retournement.")
    st.divider()
    st.subheader("Points pivots classiques — dernière bougie chargée")
    st.dataframe(pd.DataFrame({"Niveau": list(pivots.keys()), "Valeur": list(pivots.values())}).style.format({"Valeur":"{:.4f}"}), hide_index=True, use_container_width=True)
    st.subheader("Market Profile / Volume Profile (approximation)")
    if profile:
        p1,p2,p3 = st.columns(3)
        p1.metric("POC estimé", f"{profile['poc']:,.4f}")
        p2.metric("VAH estimé", f"{profile['vah']:,.4f}")
        p3.metric("VAL estimé", f"{profile['val']:,.4f}")
        vp = go.Figure(go.Bar(x=profile["volumes"], y=profile["centers"], orientation="h"))
        vp.update_layout(template="plotly_dark", height=350, title="Volume par tranche de prix (approximation)", xaxis_title="Volume", yaxis_title="Prix")
        st.plotly_chart(vp, use_container_width=True)
        st.caption("Le volume de chaque bougie est affecté à son prix typique (Haut + Bas + Clôture)/3. Ce n'est pas un profil tick par tick.")
    else:
        st.info("Profil de volume indisponible pour ces données.")

with tab_risk:
    st.subheader("Calculateur de position")
    c1,c2,c3 = st.columns(3)
    with c1:
        capital = st.number_input("Capital du compte (€)", min_value=0.0, value=10000.0, step=500.0)
        risk_pct = st.number_input("Risque par trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    with c2:
        entry = st.number_input("Prix d'entrée", min_value=0.000001, value=float(last.close), format="%.6f")
        stop = st.number_input("Stop-loss", min_value=0.000001, value=float(last.close - (last.atr if pd.notna(last.atr) else last.close*0.02)), format="%.6f")
    with c3:
        target = st.number_input("Objectif / take-profit", min_value=0.000001, value=float(last.close + (2*(last.atr if pd.notna(last.atr) else last.close*0.02))), format="%.6f")
        direction = st.selectbox("Sens", ["Achat (long)", "Vente (short)"])
    risk_cash = capital*risk_pct/100
    valid = (entry > stop and target > entry) if direction.startswith("Achat") else (stop > entry and target < entry)
    if valid:
        distance = abs(entry-stop)
        qty = risk_cash/distance if distance else 0
        potential = abs(target-entry)*qty
        rr = abs(target-entry)/distance if distance else 0
        r1,r2,r3 = st.columns(3)
        r1.metric("Risque monétaire", f"{risk_cash:,.2f} €")
        r2.metric("Quantité théorique", f"{qty:,.4f}")
        r3.metric("Ratio rendement / risque", f"1 : {rr:.2f}")
        st.write(f"Gain théorique à l'objectif, hors frais : **{potential:,.2f} €**")
    else:
        st.error("Niveaux incohérents avec le sens choisi. Vérifie entrée, stop et objectif.")
    st.caption("Calcul simplifié : il ne tient pas compte des frais, du spread, du slippage, des tailles de contrat, de l'effet de levier ni du risque de change. Vérifie les spécifications de ton courtier.")

with tab_journal:
    st.subheader("Journal de trading")
    st.caption("Les entrées sont conservées dans la session du navigateur. Télécharge régulièrement le CSV pour garder une copie.")
    if "trades" not in st.session_state:
        st.session_state.trades = pd.DataFrame(columns=["Date","Instrument","Sens","Entrée","Sortie","Résultat (€)","Respect du plan","Commentaire"])
    with st.form("trade_form", clear_on_submit=True):
        a,b,c = st.columns(3)
        trade_date = a.date_input("Date", value=datetime.now().date())
        trade_symbol = b.text_input("Instrument", value=ticker)
        trade_side = c.selectbox("Sens", ["Long","Short"])
        d,e,f = st.columns(3)
        trade_entry = d.number_input("Entrée", min_value=0.0, value=float(last.close))
        trade_exit = e.number_input("Sortie", min_value=0.0, value=float(last.close))
        trade_result = f.number_input("Résultat (€)", value=0.0, step=1.0)
        plan = st.selectbox("Plan respecté ?", ["Oui","Non","Partiellement"])
        comment = st.text_input("Commentaire / émotion / erreur")
        submitted = st.form_submit_button("Ajouter au journal")
        if submitted:
            new = pd.DataFrame([{"Date":str(trade_date),"Instrument":trade_symbol,"Sens":trade_side,"Entrée":trade_entry,"Sortie":trade_exit,"Résultat (€)":trade_result,"Respect du plan":plan,"Commentaire":comment}])
            st.session_state.trades = pd.concat([st.session_state.trades,new],ignore_index=True)
    trades = st.session_state.trades
    if not trades.empty:
        st.dataframe(trades, use_container_width=True, hide_index=True)
        st.metric("Résultat cumulé enregistré", f"{pd.to_numeric(trades['Résultat (€)']).sum():,.2f} €")
        st.download_button("Télécharger mon journal (CSV)", trades.to_csv(index=False).encode("utf-8-sig"), file_name="journal_trading.csv", mime="text/csv")
    else:
        st.info("Ton journal est vide. Ajoute ton premier trade après l'avoir clôturé.")

with tab_method:
    st.subheader("Sources et limites")
    st.markdown("""
    - **Données** : Yahoo Finance via la bibliothèque yfinance. Ce service gratuit n'est pas une API de trading garantie ; couverture, délais, historique et disponibilité varient.
    - **VWAP** : calcul cumulatif sur la période téléchargée. Pour l'intraday, il n'est pas réinitialisé automatiquement à chaque séance dans cette V1.
    - **POC / VAH / VAL** : estimation à partir du volume agrégé par bougie et de son prix typique, pas un vrai profil basé sur les transactions individuelles.
    - **Supports / résistances** : pivots locaux calculés sur une fenêtre glissante ; niveaux indicatifs.
    - **Dow** : la lecture de tendance est simplifiée via SMA et pivots ; ce n'est pas une analyse complète de la théorie de Dow.
    - **Signaux** : score de confluence heuristique, sans validation statistique ni backtest.
    - **Scalping** : les intervalles intraday peuvent être limités par Yahoo Finance ; latence et qualité des données inadaptées à une exécution rapide.
    - **Journal** : données en session, non garanties persistantes sur un hébergement gratuit. Export CSV recommandé.
    """)
    st.info("Cette plateforme est un outil pédagogique d'analyse. Elle ne passe aucun ordre et ne garantit aucun résultat financier.")
    st.write("Version : 1.0 — construit pour une première mise en ligne gratuite.")
