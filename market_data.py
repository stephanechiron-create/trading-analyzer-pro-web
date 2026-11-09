"""Adaptateur EODHD REST. Clé API via variable EODHD_API_KEY ou Streamlit Secrets."""
from datetime import datetime, timedelta, timezone
import os
import pandas as pd
import requests
BASE="https://eodhd.com/api"
EODHD_MAP={
 "AAPL":"AAPL.US","NVDA":"NVDA.US","MSFT":"MSFT.US","AMZN":"AMZN.US","GOOGL":"GOOGL.US","META":"META.US","TSLA":"TSLA.US",
 "AIR.PA":"AIR.PA","TTE.PA":"TTE.PA","MC.PA":"MC.PA","SAN.PA":"SAN.PA",
 "EURUSD=X":"EURUSD.FOREX","GBPUSD=X":"GBPUSD.FOREX","JPY=X":"USDJPY.FOREX","AUDUSD=X":"AUDUSD.FOREX",
 "CHF=X":"USDCHF.FOREX","CAD=X":"USDCAD.FOREX",
 "BTC-USD":"BTC-USD.CC","ETH-USD":"ETH-USD.CC","SOL-USD":"SOL-USD.CC","XRP-USD":"XRP-USD.CC","BNB-USD":"BNB-USD.CC","DOGE-USD":"DOGE-USD.CC"}
def _key():
 k=os.getenv("EODHD_API_KEY","").strip()
 if k:return k
 try:
  import streamlit as st
  return str(st.secrets.get("EODHD_API_KEY","")).strip()
 except Exception:return ""
def provider_status():
 return "EODHD : clé API détectée ; disponibilité selon forfait et mapping." if _key() else "EODHD sélectionné mais clé API absente. Ajoute EODHD_API_KEY dans Streamlit Secrets ou les variables Windows. Pas de bascule silencieuse vers Yahoo."
def _get(path,params):
 if not _key():return None
 q=dict(params);q.update(api_token=_key(),fmt="json")
 try:
  r=requests.get(f"{BASE}/{path}",params=q,timeout=20);r.raise_for_status();data=r.json()
  return data if isinstance(data,list) else None
 except (requests.RequestException,ValueError):return None
def _frame(rows):
 if not rows:return pd.DataFrame()
 d=pd.DataFrame(rows).rename(columns={"datetime":"date"})
 if "date" not in d and "timestamp" in d:d["date"]=pd.to_datetime(d.timestamp,unit="s",utc=True)
 if "date" not in d:return pd.DataFrame()
 need=["open","high","low","close"]
 if not all(c in d for c in need):return pd.DataFrame()
 if "volume" not in d:d["volume"]=0
 d.index=pd.to_datetime(d.date,utc=True,errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
 for c in need+["volume"]:d[c]=pd.to_numeric(d[c],errors="coerce")
 return d[need+["volume"]].dropna(subset=need).sort_index()
def fetch_eodhd(ticker,period,interval,asset_class="Actions"):
 sym=EODHD_MAP.get(str(ticker).strip().upper())
 if not sym or not _key():return pd.DataFrame()
 now=datetime.now(timezone.utc)
 if interval in ("1d","1wk"):
  days={"5d":10,"1mo":45,"3mo":120,"6mo":240,"1y":400,"2y":800,"5y":1900}.get(period,400)
  rows=_get(f"eod/{sym}",{"from":(now-timedelta(days=days)).date().isoformat(),"to":now.date().isoformat(),"period":"d","order":"a"})
  d=_frame(rows)
  if interval=="1wk" and not d.empty:d=d.resample("W-FRI").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
  return d
 if interval not in ("1m","5m","1h"):return pd.DataFrame()
 days={"5d":8,"1mo":40,"3mo":100,"6mo":180}.get(period,40)
 rows=_get(f"intraday/{sym}",{"interval":interval,"from":int((now-timedelta(days=days)).timestamp()),"to":int(now.timestamp()),"order":"a"})
 return _frame(rows)
