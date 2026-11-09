import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
import os
from market_data import fetch_eodhd, provider_status

st.set_page_config(page_title="Trading Analyzer Pro V3", page_icon="📈", layout="wide")
st.markdown("""
<style>
.block-container {padding-top:1.1rem; padding-bottom:2rem; max-width:1600px;}
[data-testid="stMetric"] {background:rgba(120,120,120,.07);padding:10px;border-radius:10px;}
.small-note {font-size:.85rem;opacity:.8}
</style>
""", unsafe_allow_html=True)

VERSION = "3.1"
PRESETS = {
    "Actions": {"Apple":"AAPL","Nvidia":"NVDA","Microsoft":"MSFT","TotalEnergies":"TTE.PA","Airbus":"AIR.PA"},
    "Indices": {"Nasdaq 100":"^NDX","S&P 500":"^GSPC","CAC 40":"^FCHI","DAX":"^GDAXI","Dow Jones":"^DJI"},
    "Forex": {"EUR/USD":"EURUSD=X","GBP/USD":"GBPUSD=X","USD/JPY":"JPY=X","AUD/USD":"AUDUSD=X"},
    "Crypto": {"Bitcoin":"BTC-USD","Ethereum":"ETH-USD","Solana":"SOL-USD"},
    "Matières premières": {"Or":"GC=F","Argent":"SI=F","Pétrole WTI":"CL=F","Gaz naturel":"NG=F"},
}
TF_CONFIG = {
    "5 min": ("5d","5m"),
    "15 min": ("1mo","15m"),
    "1 heure": ("3mo","1h"),
    "4 heures": ("6mo","1h"),  # Yahoo ne fournit pas toujours le 4h : resampling depuis le 1h.
    "Journalier": ("2y","1d"),
    "Hebdomadaire": ("5y","1wk"),
}
TF_ORDER = ["5 min","15 min","1 heure","4 heures","Journalier","Hebdomadaire"]
TRADING_STYLES = {
    "Scalping": {"timeframes": ["5 min", "15 min"], "description": "Très court terme : nécessite des données fraîches, un spread faible et une exécution rapide.", "risk_note": "Le flux REST historique seul ne suffit pas pour du scalping en réel."},
    "Day trading": {"timeframes": ["5 min", "15 min", "1 heure"], "description": "Positions ouvertes et clôturées dans la même séance.", "risk_note": "Vérifier horaires de marché, spread et annonces économiques."},
    "Swing trading": {"timeframes": ["4 heures", "Journalier", "Hebdomadaire"], "description": "Positions conservées plusieurs jours ou semaines.", "risk_note": "Confirmer les signaux à la clôture de la bougie."},
}


@st.cache_data(ttl=60, show_spinner=False)
def fetch_yahoo_data(ticker, period, interval):
    try:
        raw = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
        if raw is None or raw.empty: return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
        raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
        cols = ["open", "high", "low", "close", "volume"]
        if not all(c in raw.columns for c in cols): return pd.DataFrame()
        out = raw[cols].copy().dropna(subset=["open", "high", "low", "close"])
        out.index = pd.to_datetime(out.index)
        if getattr(out.index, "tz", None) is not None: out.index = out.index.tz_convert("UTC").tz_localize(None)
        return out
    except Exception: return pd.DataFrame()

def fetch_data(ticker, period, interval, provider="Yahoo Finance", asset_class="Actions"):
    if provider == "EODHD": return fetch_eodhd(ticker, period, interval, asset_class)
    return fetch_yahoo_data(ticker, period, interval)

def get_tf_data(ticker, tf, provider="Yahoo Finance", asset_class="Actions"):
    period, interval = TF_CONFIG[tf]
    if provider == "EODHD":
        if tf == "5 min": return fetch_data(ticker, "5d", "5m", provider, asset_class)
        if tf == "15 min":
            base = fetch_data(ticker, "5d", "5m", provider, asset_class)
            return base.resample("15min", origin="start_day").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna() if not base.empty else base
        if tf == "1 heure": return fetch_data(ticker, "3mo", "1h", provider, asset_class)
        if tf == "4 heures":
            base = fetch_data(ticker, "6mo", "1h", provider, asset_class)
            return base.resample("4h", origin="start_day").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna() if not base.empty else base
        if tf == "Journalier": return fetch_data(ticker, "2y", "1d", provider, asset_class)
        base = fetch_data(ticker, "5y", "1d", provider, asset_class)
        return base.resample("W-FRI").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna() if not base.empty else base
    df = fetch_data(ticker, period, interval, provider, asset_class)
    if tf == "4 heures" and not df.empty:
        df = df.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    return df

def indicators(df):
    x = df.copy()
    c = x.close
    x["sma20"] = c.rolling(20).mean()
    x["sma50"] = c.rolling(50).mean()
    x["sma200"] = c.rolling(200).mean()
    x["ema12"] = c.ewm(span=12, adjust=False).mean()
    x["ema26"] = c.ewm(span=26, adjust=False).mean()
    x["macd"] = x.ema12 - x.ema26
    x["macd_signal"] = x.macd.ewm(span=9, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi"] = 100 - 100/(1+rs)
    prev = c.shift(1)
    tr = pd.concat([(x.high-x.low),(x.high-prev).abs(),(x.low-prev).abs()],axis=1).max(axis=1)
    x["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    x["bb_mid"], x["bb_upper"], x["bb_lower"] = mid, mid+2*sd, mid-2*sd
    typical = (x.high+x.low+c)/3
    vol = x.volume.fillna(0)
    # Intraday VWAP resets each session; for daily/weekly, cumulative over loaded history.
    if len(x.index) and (x.index[-1]-x.index[0]).total_seconds() < 90*86400 and x.index[-1].hour != 0:
        session = pd.Series(x.index.date, index=x.index)
        pv = typical*vol
        x["vwap"] = pv.groupby(session).cumsum()/vol.groupby(session).cumsum().replace(0,np.nan)
    else:
        x["vwap"] = (typical*vol).cumsum()/vol.cumsum().replace(0,np.nan)
    return x

def trend_reading(df):
    if len(df) < 55:
        return "Données insuffisantes", 0
    r = df.iloc[-1]
    if pd.isna(r.sma20) or pd.isna(r.sma50):
        return "Données insuffisantes", 0
    score = int(r.sma20 > r.sma50) + int(r.close > r.sma20)
    score -= int(r.sma20 < r.sma50) + int(r.close < r.sma20)
    if score >= 2: return "Haussière", score
    if score <= -2: return "Baissière", score
    return "Neutre / mixte", score

def candle_patterns(df):
    if len(df)<2: return []
    a,b=df.iloc[-2],df.iloc[-1]
    body=abs(b.close-b.open); rng=max(b.high-b.low,1e-12)
    lower=min(b.open,b.close)-b.low; upper=b.high-max(b.open,b.close)
    out=[]
    if body/rng<.1: out.append("Doji")
    if lower>body*2 and upper<body*.8 and body/rng<.4: out.append("Marteau potentiel")
    if b.close>a.open and b.open<a.close and a.close<a.open and b.close>b.open: out.append("Avalement haussier")
    if b.close<a.open and b.open>a.close and a.close>a.open and b.close<b.open: out.append("Avalement baissier")
    return out

def pivots(df):
    pbar = df.iloc[-2] if len(df)>1 else df.iloc[-1]
    p=(pbar.high+pbar.low+pbar.close)/3
    span=pbar.high-pbar.low
    return {"Pivot":p,"R1":2*p-pbar.low,"S1":2*p-pbar.high,"R2":p+span,"S2":p-span}

def swing_levels(df, window=3):
    highs=[]; lows=[]
    h=df.high.to_numpy(); l=df.low.to_numpy()
    for i in range(window,len(df)-window):
        if h[i] == np.max(h[i-window:i+window+1]): highs.append(float(h[i]))
        if l[i] == np.min(l[i-window:i+window+1]): lows.append(float(l[i]))
    def compact(vals):
        vals=sorted(vals); out=[]
        for v in vals:
            if not out or abs(v-out[-1])/max(abs(v),1e-9)>.003: out.append(v)
        return out[-5:]
    return compact(lows),compact(highs)

def volume_profile(df, bins=32):
    if df.empty: return None
    typical=(df.high+df.low+df.close)/3
    lo,hi=float(df.low.min()),float(df.high.max())
    if not np.isfinite(lo+hi) or hi<=lo: return None
    edges=np.linspace(lo,hi,bins+1)
    ids=np.clip(np.digitize(typical,edges)-1,0,bins-1)
    vols=np.zeros(bins)
    for i,v in zip(ids,df.volume.fillna(0).to_numpy()): vols[i]+=max(float(v),0)
    if vols.sum()<=0: return None
    centers=(edges[:-1]+edges[1:])/2
    poc=int(np.argmax(vols)); chosen={poc}; total=vols[poc]; target=.70*vols.sum()
    left,right=poc-1,poc+1
    while total<target and (left>=0 or right<bins):
        lv=vols[left] if left>=0 else -1
        rv=vols[right] if right<bins else -1
        if rv>lv: chosen.add(right); total+=max(rv,0); right+=1
        else: chosen.add(left); total+=max(lv,0); left-=1
    return {"poc":float(centers[poc]),"vah":float(centers[max(chosen)]),
            "val":float(centers[min(chosen)]),"centers":centers,"volumes":vols}

def make_signal(frames):
    available=[(tf,df) for tf,df in frames.items() if len(df)>=55]
    if not available: return "ATTENTE", "Historique insuffisant sur les unités sélectionnées.", 0
    scores=[]
    for tf,df in available:
        _,s=trend_reading(df); scores.append((tf,s))
    total=sum(s for _,s in scores)
    details=[f"{tf}: {('haussier' if s>0 else 'baissier' if s<0 else 'neutre')} ({s:+d})" for tf,s in scores]
    # Signal uses majority alignment, but a single lower timeframe cannot overrule the higher ones.
    higher=[s for tf,s in scores if tf in ("1 heure","4 heures","Journalier","Hebdomadaire")]
    if not higher: higher=[s for _,s in scores]
    high=sum(higher)
    if high>=2 and total>=3: label="ACHAT À ÉTUDIER"
    elif high<=-2 and total<=-3: label="VENTE À ÉTUDIER"
    else: label="ATTENTE"
    return label, "Confluence multi-unités : "+" ; ".join(details), total

def backtest_sma(df, fee_bps=5, slippage_bps=2, mode="Long uniquement"):
    # Signal at close t is shifted to exposure during t+1: no same-bar look-ahead.
    x=df.copy()
    x["fast"]=x.close.rolling(20).mean()
    x["slow"]=x.close.rolling(50).mean()
    raw=np.where(x.fast>x.slow,1,0)
    if mode=="Long / short": raw=np.where(x.fast>x.slow,1,-1)
    pos=pd.Series(raw,index=x.index).shift(1).fillna(0)
    asset_ret=x.close.pct_change().fillna(0)
    turnover=pos.diff().abs().fillna(pos.abs())
    cost=(fee_bps+slippage_bps)/10000
    strat_ret=pos*asset_ret-turnover*cost
    equity=(1+strat_ret).cumprod()
    peak=equity.cummax()
    dd=equity/peak-1
    changes=pos.diff().fillna(pos).abs()
    trade_entries=int((changes>0).sum())
    wins=int((strat_ret[changes>0]>0).sum())
    gross_profit=float(strat_ret[strat_ret>0].sum())
    gross_loss=float(abs(strat_ret[strat_ret<0].sum()))
    pf=gross_profit/gross_loss if gross_loss>0 else np.nan
    return {"equity":equity,"returns":strat_ret,"total_return":float(equity.iloc[-1]-1),
            "max_dd":float(dd.min()),"win_rate":wins/max(trade_entries,1),
            "profit_factor":pf,"trades":trade_entries}


# Univers de surveillance multi-marchés. Liste volontairement limitée et modifiable.
SCAN_UNIVERSE = {
    "Actions": {
        "Apple": "AAPL", "Nvidia": "NVDA", "Microsoft": "MSFT",
        "Amazon": "AMZN", "Alphabet": "GOOGL", "Meta": "META",
        "Airbus": "AIR.PA", "TotalEnergies": "TTE.PA",
        "LVMH": "MC.PA", "Sanofi": "SAN.PA"
    },
    "Indices": {
        "Nasdaq 100": "^NDX", "S&P 500": "^GSPC", "CAC 40": "^FCHI",
        "DAX": "^GDAXI", "Dow Jones": "^DJI", "Euro Stoxx 50": "^STOXX50E"
    },
    "Forex": {
        "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X",
        "USD/JPY": "JPY=X", "AUD/USD": "AUDUSD=X",
        "USD/CHF": "CHF=X", "USD/CAD": "CAD=X"
    },
    "Cryptomonnaies": {
        "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD",
        "Solana": "SOL-USD", "XRP": "XRP-USD",
        "BNB": "BNB-USD", "Dogecoin": "DOGE-USD"
    },
    "Matières premières": {
        "Or": "GC=F", "Argent": "SI=F", "Pétrole WTI": "CL=F",
        "Gaz naturel": "NG=F", "Cuivre": "HG=F", "Platine": "PL=F"
    }
}

def scan_one_asset(name, symbol, tf_list, provider="Yahoo Finance", asset_class="Actions"):
    """Analyse technique indicative d'un actif sur les unités choisies."""
    rows = []
    usable = []
    for tf in tf_list:
        raw = get_tf_data(symbol, tf, provider, asset_class)
        if raw.empty or len(raw) < 55:
            continue
        d = indicators(raw)
        r = d.iloc[-1]
        prev = d.iloc[-2]
        trend, tscore = trend_reading(d)
        close = float(r.close)
        atr = float(r.atr) if pd.notna(r.atr) else np.nan
        rsi = float(r.rsi) if pd.notna(r.rsi) else np.nan
        macd_bull = bool(pd.notna(r.macd) and pd.notna(r.macd_signal) and r.macd > r.macd_signal)
        vol_mean = d.volume.rolling(20).mean().iloc[-1]
        vol_ratio = float(r.volume / vol_mean) if pd.notna(vol_mean) and vol_mean > 0 else np.nan
        above_vwap = bool(pd.notna(r.vwap) and close > float(r.vwap))
        bb_width = ((float(r.bb_upper-r.bb_lower)/close)*100
                    if pd.notna(r.bb_upper) and pd.notna(r.bb_lower) and close else np.nan)
        bb_width_median = ((d.bb_upper-d.bb_lower)/d.close.replace(0,np.nan)).rolling(60).median().iloc[-1]
        squeeze = bool(pd.notna(bb_width_median) and pd.notna(r.bb_upper) and
                       ((r.bb_upper-r.bb_lower)/close) <= bb_width_median)
        recent_high = float(d.high.iloc[-21:-1].max()) if len(d) >= 22 else float(d.high.iloc[:-1].max())
        recent_low = float(d.low.iloc[-21:-1].min()) if len(d) >= 22 else float(d.low.iloc[:-1].min())
        breakout_up = close > recent_high
        breakout_down = close < recent_low

        score = 0
        score += 20 if tscore == 2 else 10 if tscore == 1 else -15 if tscore == -2 else -7 if tscore == -1 else 0
        score += 10 if macd_bull else -5
        score += 10 if above_vwap else -5
        score += 10 if 45 <= rsi <= 65 else 5 if 35 <= rsi < 45 or 65 < rsi <= 70 else -8 if rsi > 75 or rsi < 25 else 0
        score += 15 if breakout_up else -10 if breakout_down else 0
        score += 10 if pd.notna(vol_ratio) and vol_ratio >= 1.5 else 0
        score += 8 if squeeze else 0
        score = max(0, min(100, score + 25))
        if breakout_up and macd_bull:
            setup = "Cassure haussière à confirmer"
        elif breakout_down:
            setup = "Rupture baissière à confirmer"
        elif squeeze:
            setup = "Compression — surveiller la sortie"
        elif trend == "Haussière" and 40 <= rsi <= 62:
            setup = "Tendance haussière / repli possible"
        elif trend == "Baissière" and 38 <= rsi <= 60:
            setup = "Tendance baissière / rebond à risque"
        elif rsi >= 72:
            setup = "Surachat — éviter de poursuivre"
        elif rsi <= 28:
            setup = "Survente — retournement non confirmé"
        else:
            setup = "Configuration neutre"
        direction = "Long" if score >= 60 else "Short" if score <= 30 else "À attendre"
        # Niveaux indicatifs calculés à partir du dernier cours et de l'ATR.
        if np.isfinite(atr) and atr > 0 and direction == "Long":
            entry = close
            stop = close - 1.5 * atr
            target1 = close + 2 * (entry-stop)
            target2 = close + 3 * (entry-stop)
        elif np.isfinite(atr) and atr > 0 and direction == "Short":
            entry = close
            stop = close + 1.5 * atr
            target1 = close - 2 * (stop-entry)
            target2 = close - 3 * (stop-entry)
        else:
            entry = stop = target1 = target2 = np.nan
        rows.append({
            "Instrument": name, "Ticker": symbol, "Unité": tf,
            "Cours": close, "Tendance": trend, "RSI": rsi,
            "Score technique /100": score, "Configuration": setup,
            "Orientation": direction, "Volume x moyenne": vol_ratio,
            "Entrée théorique": entry, "Stop-loss": stop,
            "Objectif 1 (2R)": target1, "Objectif 2 (3R)": target2,
            "Ratio cible": "1:2 / 1:3" if direction in ("Long","Short") else "—"
        })
        usable.append((tf, score))
    return rows


st.title("📈 Trading Analyzer Pro")
st.caption("V3.1 • Profils scalping / day trading / swing • Fournisseur EODHD ou Yahoo Finance")
st.caption("Version 3.1 • Analyse multi-unités, scanner multi-marchés, gestion du risque et backtesting pédagogique")

with st.sidebar:
    st.header("Marché")
    category=st.selectbox("Catégorie",list(PRESETS))
    data_provider = st.selectbox("Fournisseur de données", ["Yahoo Finance", "EODHD"], index=0)
    if data_provider == "EODHD": st.caption("EODHD nécessite une clé API et un symbole pris en charge.")
    trading_style = st.selectbox("Style de trading", list(TRADING_STYLES), index=2)
    st.caption(TRADING_STYLES[trading_style]["description"])
    instrument=st.selectbox("Instrument",list(PRESETS[category]))
    ticker=st.text_input("Ticker / symbole",value=PRESETS[category][instrument]).strip().upper()
    st.divider()
    st.subheader("Unités de temps")
    selected_tfs=st.multiselect("Unités à comparer",TF_ORDER,default=TF_ORDER)
    chart_tf=st.selectbox("Unité du graphique principal",selected_tfs or TF_ORDER,index=0)
    refresh=st.button("↻ Actualiser les données",use_container_width=True)
    if refresh: st.cache_data.clear()
    st.caption("Les données intraday gratuites sont limitées. Le 4 h est reconstruit à partir des bougies 1 h.")

if not ticker:
    st.error("Saisis un ticker valide."); st.stop()
if not selected_tfs:
    st.warning("Sélectionne au moins une unité de temps."); st.stop()
if data_provider == "EODHD": st.caption(provider_status())

frames={}
with st.spinner(f"Chargement des unités de temps de {ticker}…"):
    for tf in selected_tfs:
        raw=get_tf_data(ticker,tf,data_provider,category)
        if not raw.empty: frames[tf]=indicators(raw)

if chart_tf not in frames:
    st.error(f"Aucune donnée disponible pour {ticker} en {chart_tf}. Essaie une autre unité ou un autre actif.")
    st.stop()
if len(frames[chart_tf])<5:
    st.error("Historique trop court pour afficher l'analyse."); st.stop()

df=frames[chart_tf]; last=df.iloc[-1]; prev=df.iloc[-2]
change=(last.close/prev.close-1)*100 if prev.close else 0
signal,signal_reason,confluence=make_signal(frames)
trend,trend_score=trend_reading(df)
profile=volume_profile(df)
supports,resistances=swing_levels(df)
piv=pivots(df)
patterns=candle_patterns(df)

m1,m2,m3,m4,m5=st.columns(5)
m1.metric("Dernier cours",f"{last.close:,.4f}",f"{change:+.2f}%")
m2.metric("RSI (14)",f"{last.rsi:.1f}" if pd.notna(last.rsi) else "—")
m3.metric("ATR (14)",f"{last.atr:.4f}" if pd.notna(last.atr) else "—")
m4.metric("Tendance du graphique",trend)
m5.metric("Confluence multi-unités",f"{confluence:+d}")

if signal=="ACHAT À ÉTUDIER": st.success("Signal indicatif : ACHAT À ÉTUDIER")
elif signal=="VENTE À ÉTUDIER": st.error("Signal indicatif : VENTE À ÉTUDIER")
else: st.info("Signal indicatif : ATTENTE")
st.caption(signal_reason)

tab1,tab2,tab3,tab4,tab5,tab6=st.tabs(["📊 Graphique","🧭 Multi-unités & niveaux","🛡️ Gestion du risque","🧪 Backtesting","📓 Journal & méthode","🔎 Opportunités à surveiller"])

with tab1:
    overlays=st.multiselect("Indicateurs sur le graphique",["SMA 20","SMA 50","SMA 200","Bandes de Bollinger","VWAP"],default=["SMA 20","SMA 50","Bandes de Bollinger"])
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.04,row_heights=[.75,.25])
    fig.add_trace(go.Candlestick(x=df.index,open=df.open,high=df.high,low=df.low,close=df.close,name="Prix"),row=1,col=1)
    for name,col in [("SMA 20","sma20"),("SMA 50","sma50"),("SMA 200","sma200"),("VWAP","vwap")]:
        if name in overlays: fig.add_trace(go.Scatter(x=df.index,y=df[col],name=name,line=dict(width=1.4)),row=1,col=1)
    if "Bandes de Bollinger" in overlays:
        fig.add_trace(go.Scatter(x=df.index,y=df.bb_upper,name="Bollinger haute",line=dict(width=1,dash="dot")),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df.bb_lower,name="Bollinger basse",line=dict(width=1,dash="dot"),fill="tonexty",fillcolor="rgba(100,130,180,.08)"),row=1,col=1)
    fig.add_trace(go.Bar(x=df.index,y=df.volume,name="Volume",opacity=.55),row=2,col=1)
    fig.update_layout(height=680,template="plotly_dark",xaxis_rangeslider_visible=False,margin=dict(l=10,r=10,t=30,b=10),legend=dict(orientation="h"))
    fig.update_yaxes(title_text="Prix",row=1,col=1); fig.update_yaxes(title_text="Volume",row=2,col=1)
    st.plotly_chart(fig,use_container_width=True)
    st.caption(f"{len(df)} bougies • Dernière bougie : {df.index[-1]} UTC • Unité : {chart_tf} • Source : {data_provider}")
    st.write("Figures de la dernière bougie :",", ".join(patterns) if patterns else "Aucune figure simple détectée.")

with tab2:
    st.subheader("Tableau de bord multi-unités")
    rows=[]
    for tf in TF_ORDER:
        if tf in frames:
            d=frames[tf]; tr,sc=trend_reading(d); r=d.iloc[-1]
            rows.append({"Unité":tf,"Tendance":tr,"Score":sc,"RSI":round(float(r.rsi),1) if pd.notna(r.rsi) else None,
                         "Cours":float(r.close),"SMA20":float(r.sma20) if pd.notna(r.sma20) else None,
                         "SMA50":float(r.sma50) if pd.notna(r.sma50) else None,"Bougies":len(d)})
        else: rows.append({"Unité":tf,"Tendance":"Indisponible","Score":None,"RSI":None,"Cours":None,"SMA20":None,"SMA50":None,"Bougies":0})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    st.subheader("Supports et résistances — unité du graphique")
    a,b=st.columns(2)
    a.write("**Supports**"); a.write(" · ".join(f"{v:,.4f}" for v in supports) if supports else "Insuffisants")
    b.write("**Résistances**"); b.write(" · ".join(f"{v:,.4f}" for v in resistances) if resistances else "Insuffisantes")
    st.subheader("Points pivots (calculés avec la bougie précédente)")
    st.dataframe(pd.DataFrame({"Niveau":list(piv),"Valeur":list(piv.values())}).style.format({"Valeur":"{:.4f}"}),hide_index=True,use_container_width=True)
    st.subheader("Volume Profile estimatif")
    if profile:
        c1,c2,c3=st.columns(3)
        c1.metric("POC estimé",f"{profile['poc']:,.4f}"); c2.metric("VAH estimé",f"{profile['vah']:,.4f}"); c3.metric("VAL estimé",f"{profile['val']:,.4f}")
        vfig=go.Figure(go.Bar(x=profile["volumes"],y=profile["centers"],orientation="h"))
        vfig.update_layout(template="plotly_dark",height=360,title="Volume par tranche de prix — approximation",xaxis_title="Volume",yaxis_title="Prix")
        st.plotly_chart(vfig,use_container_width=True)
        st.caption("Le volume de chaque bougie est affecté à son prix typique. Ce n'est pas un profil tick par tick.")
    else: st.info("Profil de volume indisponible.")

with tab3:
    st.subheader("Calculateur de position")
    c1,c2,c3=st.columns(3)
    capital=c1.number_input("Capital du compte (€)",min_value=0.0,value=10000.0,step=500.0)
    risk_pct=c1.number_input("Risque par trade (%)",min_value=.1,max_value=10.0,value=1.0,step=.1)
    direction=c1.selectbox("Sens",["Achat (long)","Vente (short)"])
    entry=c2.number_input("Prix d'entrée",min_value=.000001,value=float(last.close),format="%.6f")
    atr=float(last.atr) if pd.notna(last.atr) else max(float(last.close)*.02,.000001)
    default_stop=entry-1.5*atr if direction.startswith("Achat") else entry+1.5*atr
    stop=c2.number_input("Stop-loss",min_value=.000001,value=float(max(default_stop,.000001)),format="%.6f")
    target=c3.number_input("Objectif",min_value=.000001,value=float(max(entry+3*atr if direction.startswith("Achat") else entry-3*atr,.000001)),format="%.6f")
    fee=c3.number_input("Frais estimés aller-retour (%)",min_value=0.0,max_value=5.0,value=.10,step=.05)
    valid=(entry>stop and target>entry) if direction.startswith("Achat") else (stop>entry and target<entry)
    if valid:
        risk_cash=capital*risk_pct/100; distance=abs(entry-stop)
        qty=risk_cash/distance if distance else 0
        rr=abs(target-entry)/distance if distance else 0
        gross=abs(target-entry)*qty; net=gross-capital*fee/100
        x1,x2,x3,x4=st.columns(4)
        x1.metric("Budget de risque",f"{risk_cash:,.2f} €")
        x2.metric("Quantité théorique",f"{qty:,.4f}")
        x3.metric("Ratio gain/risque",f"1 : {rr:.2f}")
        x4.metric("Gain estimé après frais saisis",f"{net:,.2f} €")
        st.caption("Quantité théorique en unités simples. Vérifie les contrats, lots, devises, spread, levier et frais du courtier.")
    else: st.error("Les niveaux entrée / stop / objectif sont incohérents avec le sens choisi.")

with tab4:
    st.subheader("Backtesting — croisement SMA 20 / SMA 50")
    st.warning("Test pédagogique d'une stratégie simple. Ce résultat historique ne prédit pas les performances futures.")
    bt_tf=st.selectbox("Unité testée",list(frames),index=list(frames).index(chart_tf))
    bt_mode=st.selectbox("Sens de la stratégie",["Long uniquement","Long / short"])
    fee_bps=st.number_input("Frais par changement de position (points de base)",min_value=0.0,max_value=100.0,value=5.0,step=1.0)
    slip_bps=st.number_input("Slippage estimé (points de base)",min_value=0.0,max_value=100.0,value=2.0,step=1.0)
    btdf=frames[bt_tf]
    if len(btdf)<60:
        st.info("Il faut au moins 60 bougies pour un test raisonnable sur cette unité.")
    else:
        result=backtest_sma(btdf,fee_bps,slip_bps,bt_mode)
        q1,q2,q3,q4,q5=st.columns(5)
        q1.metric("Performance cumulée",f"{result['total_return']*100:+.2f}%")
        q2.metric("Drawdown maximal",f"{result['max_dd']*100:.2f}%")
        q3.metric("Trades / changements",str(result["trades"]))
        q4.metric("Taux de périodes gagnantes",f"{result['win_rate']*100:.1f}%")
        q5.metric("Profit factor",f"{result['profit_factor']:.2f}" if np.isfinite(result["profit_factor"]) else "—")
        eqfig=go.Figure(go.Scatter(x=result["equity"].index,y=result["equity"],name="Capital théorique"))
        eqfig.update_layout(template="plotly_dark",height=380,title="Courbe de capital normalisée (départ = 1)")
        st.plotly_chart(eqfig,use_container_width=True)
        st.caption("Le taux affiché porte sur les changements de position détectés, pas sur un registre exhaustif de trades exécutés. Pas de spread réel ni de fiscalité.")
        st.download_button("Exporter les résultats du test (CSV)",
            pd.DataFrame({"Date":result["equity"].index,"Capital_normalise":result["equity"].values,"Rendement_strategie":result["returns"].values}).to_csv(index=False).encode("utf-8-sig"),
            file_name=f"backtest_{ticker}_{bt_tf.replace(' ','_')}.csv",mime="text/csv")

with tab5:
    st.subheader("Journal de trading")
    st.caption("Le journal est conservé dans la session en cours. Exporte régulièrement le CSV.")
    if "trades_v2" not in st.session_state:
        st.session_state.trades_v2=pd.DataFrame(columns=["Date","Ticker","Unité","Sens","Entrée","Sortie","Résultat €","Plan respecté","Émotion / commentaire"])
    with st.form("trade_form_v2",clear_on_submit=True):
        a,b,c=st.columns(3)
        date=a.date_input("Date",value=datetime.now().date())
        symbol=b.text_input("Instrument",value=ticker)
        tf=c.selectbox("Unité",TF_ORDER,index=TF_ORDER.index(chart_tf))
        d,e,f=st.columns(3)
        side=d.selectbox("Sens",["Long","Short"])
        ent=e.number_input("Entrée",min_value=0.0,value=float(last.close))
        out=f.number_input("Sortie",min_value=0.0,value=float(last.close))
        pnl=d.number_input("Résultat (€)",value=0.0,step=1.0)
        plan=st.selectbox("Plan respecté ?",["Oui","Non","Partiellement"])
        note=st.text_input("Émotion / erreur / commentaire")
        add=st.form_submit_button("Ajouter au journal")
        if add:
            new=pd.DataFrame([{"Date":str(date),"Ticker":symbol,"Unité":tf,"Sens":side,"Entrée":ent,"Sortie":out,"Résultat €":pnl,"Plan respecté":plan,"Émotion / commentaire":note}])
            st.session_state.trades_v2=pd.concat([st.session_state.trades_v2,new],ignore_index=True)
    if not st.session_state.trades_v2.empty:
        st.dataframe(st.session_state.trades_v2,use_container_width=True,hide_index=True)
        st.metric("Résultat cumulé",f"{pd.to_numeric(st.session_state.trades_v2['Résultat €']).sum():,.2f} €")
        st.download_button("Télécharger mon journal CSV",st.session_state.trades_v2.to_csv(index=False).encode("utf-8-sig"),"journal_trading_v2.csv","text/csv")
    st.divider()
    st.subheader("Méthode et limites")
    st.markdown("""
- Données : Yahoo Finance via yfinance. Couverture, délais et historiques varient selon les actifs et unités.
- 4 heures : agrégation locale de bougies 1 heure ; les limites de séance et horodatages peuvent affecter les regroupements.
- VWAP : réinitialisation journalière approximative en intraday ; calcul cumulatif sur l'historique pour les unités supérieures.
- Volume Profile : estimation par prix typique de chaque bougie, pas données transaction par transaction.
- Signaux : règles heuristiques de tendance multi-unités, sans probabilité de réussite validée.
- Scanner V3 : univers prédéfini de valeurs liquides, non exhaustif ; disponibilité et qualité des données Yahoo Finance variables. Les scores ne sont pas des probabilités.
- Plans théoriques du scanner : entrée au dernier cours connu, stop à 1,5 ATR, objectifs à 2R et 3R ; aucune validation d’exécution ou de liquidité.
- Backtesting : SMA20/SMA50 avec signal décalé d'une bougie et coûts paramétrables. Pas encore de validation walk-forward, de contrôle de sur-optimisation ni de simulation précise des contrats.
- Journal : session temporaire ; export CSV indispensable pour conserver les données.
- Aucun ordre n'est transmis à un courtier.
""")
    st.info("Outil pédagogique. Ne constitue pas un conseil financier et ne garantit aucun résultat.")
    st.caption(f"Trading Analyzer Pro — version {VERSION}")



with tab6:
    st.subheader("Scanner multi-marchés — opportunités techniques")
    st.write("Le scanner passe en revue une sélection d'actifs dans toutes les catégories et classe les configurations selon des règles techniques explicites.")
    st.warning("Les scores mesurent une confluence de critères, pas une probabilité de gain. Les niveaux sont théoriques et ne sont pas des ordres à transmettre.")
    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    scan_categories = c1.multiselect(
        "Marchés à analyser", list(SCAN_UNIVERSE.keys()),
        default=list(SCAN_UNIVERSE.keys())
    )
    scan_style = c2.selectbox("Style à privilégier", list(TRADING_STYLES), index=list(TRADING_STYLES).index(trading_style))
    scan_tfs = st.multiselect("Unités à scanner (toutes sélectionnables)", TF_ORDER, default=TRADING_STYLES[scan_style]["timeframes"])
    st.caption(TRADING_STYLES[scan_style]["risk_note"])
    min_score = c3.slider("Score minimum", 0, 100, 55, 5)
    st.caption("Pour limiter les appels de données, commence avec 1 h, 4 h, journalier et hebdomadaire. Le scan des 6 unités est possible mais plus lent.")
    run_scan = st.button("🔍 Lancer le scan de marché", type="primary", use_container_width=True)
    if run_scan:
        if not scan_categories or not scan_tfs:
            st.error("Choisis au moins un marché et une unité de temps.")
        else:
            scan_rows = []
            assets = [(cat, nm, sym) for cat in scan_categories
                      for nm, sym in SCAN_UNIVERSE[cat].items()]
            progress = st.progress(0, text="Préparation du scan…")
            for idx, (cat, nm, sym) in enumerate(assets, start=1):
                try:
                    found = scan_one_asset(nm, sym, scan_tfs, data_provider, cat)
                    for row in found:
                        row["Marché"] = cat
                        scan_rows.append(row)
                except Exception:
                    pass
                progress.progress(idx / max(len(assets), 1),
                                  text=f"Analyse de {nm} ({idx}/{len(assets)})")
            progress.empty()
            st.session_state["scan_v3_rows"] = scan_rows
            st.session_state["scan_v3_time"] = datetime.now().strftime("%d/%m/%Y à %H:%M")
    if "scan_v3_rows" in st.session_state:
        scan_df = pd.DataFrame(st.session_state["scan_v3_rows"])
        if scan_df.empty:
            st.info("Aucune donnée exploitable n'a été récupérée. Réessaie plus tard ou réduis le nombre d'unités.")
        else:
            scan_df = scan_df[scan_df["Score technique /100"] >= min_score].copy()
            scan_df = scan_df.sort_values(["Score technique /100", "Unité"], ascending=[False, True])
            st.caption(f"Dernier scan : {st.session_state.get('scan_v3_time','—')} • {len(scan_df)} configurations au-dessus du seuil.")
            if scan_df.empty:
                st.info("Aucune configuration ne dépasse le score minimum choisi. C'est aussi un résultat utile : ne pas trader est une décision.")
            else:
                st.dataframe(
                    scan_df[["Marché","Instrument","Ticker","Unité","Cours","Tendance",
                             "Score technique /100","RSI","Configuration","Orientation"]],
                    hide_index=True, use_container_width=True,
                    column_config={
                        "Cours": st.column_config.NumberColumn(format="%.5f"),
                        "RSI": st.column_config.NumberColumn(format="%.1f"),
                        "Score technique /100": st.column_config.ProgressColumn(min_value=0,max_value=100,format="%d")
                    }
                )
                st.subheader("Plan théorique des configurations les mieux classées")
                top = scan_df.head(10)
                st.dataframe(
                    top[["Marché","Instrument","Unité","Orientation","Configuration",
                         "Entrée théorique","Stop-loss","Objectif 1 (2R)","Objectif 2 (3R)","Ratio cible"]],
                    hide_index=True, use_container_width=True
                )
                csv = scan_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("Télécharger les résultats du scan (CSV)", csv,
                                   file_name="trading_analyzer_scan.csv", mime="text/csv")
            st.caption("Un signal est à examiner sur le graphique avant toute décision. Les niveaux sont basés sur l'ATR (stop à 1,5 ATR, objectifs à 2R et 3R) et ne tiennent pas compte du spread, des frais, du slippage, du levier ou des spécificités de contrat.")
    else:
        st.info("Clique sur « Lancer le scan de marché » pour rechercher des configurations dans les différentes catégories.")
