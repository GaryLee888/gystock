import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import twstock
import warnings

# --- 基礎設定 ---
st.set_page_config(page_title="台股全方位決策系統", layout="wide")
warnings.filterwarnings("ignore")

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class StockMaster:
    def __init__(self):
        self.special_mapping = {"貝爾威勒": "7861", "能率亞洲": "7777", "力旺": "3529", "朋程": "8255"}

    def fetch_data(self, sid):
        for suffix in [".TW", ".TWO"]:
            try:
                ticker = f"{sid}{suffix}"
                df = yf.download(ticker, period="1y", progress=False)
                if df is not None and not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    return df, ticker
            except: continue
        return None, None

    def calculate_indicators(self, df):
        if len(df) < 20: return None
        df = df.copy()
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        df['BB_up'] = df['MA20'] + (std * 2)
        df['BB_low'] = df['MA20'] - (std * 2)
        low_9 = df['Low'].rolling(9).min()
        high_9 = df['High'].rolling(9).max()
        df['K'] = ((df['Close'] - low_9) / (high_9 - low_9).replace(0, np.nan) * 100).ewm(com=2).mean()
        df['D'] = df['K'].ewm(com=2).mean()
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['MACD_hist'] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss).replace(0, np.nan)))
        tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        df['VMA20'] = df['Volume'].rolling(20).mean()
        return df.dropna()

# --- 側邊欄控制 ---
with st.sidebar:
    st.title("🛡️ 交易參數")
    atr_mult = st.slider("ATR 止損倍數", 1.5, 3.5, 2.2)
    reward_ratio = st.slider("盈虧比 (TP)", 1.0, 5.0, 2.0)
    st.divider()
    st.header("🔍 批次分析名單")
    default_vals = ["2330", "2317", "2454", "能率亞洲", "2603", "2881", "", "", "", ""]
    input_queries = []
    for i in range(10):
        val = st.text_input(f"股票 {i+1}", value=default_vals[i], key=f"in_{i}")
        if val: input_queries.append(val)

# --- 主畫面顯示 ---
st.title("📈 台股全方位決策系統")

if input_queries:
    master = StockMaster()
    tabs = st.tabs([f"分析: {q}" for q in input_queries])
    
    for tab, query in zip(tabs, input_queries):
        with tab:
            sid = master.special_mapping.get(query, query)
            if not sid.isdigit():
                for code, info in twstock.codes.items():
                    if query in info.name:
                        sid = code; break
            
            df_raw, ticker_str = master.fetch_data(sid)
            
            if df_raw is not None:
                df = master.calculate_indicators(df_raw)
                if df is not None:
                    curr = df.iloc[-1]
                    prev = df.iloc[-2]
                    curr_p = float(curr['Close'])
                    
                    # 計算數據
                    entry_p = float(curr['MA20'])
                    sl_p = entry_p - (float(curr['ATR']) * atr_mult)
                    tp_p = entry_p + (entry_p - sl_p) * reward_ratio
                    
                    # --- 核心資訊放在最上方 ---
                    # 1. 計算總分
                    conds = {
                        "均線趨勢": (curr_p > curr['MA20'], "多頭", "空頭"),
                        "KD動能": (curr['K'] > curr['D'], "黃金交叉", "死亡交叉"),
                        "MACD柱狀": (curr['MACD_hist'] > 0, "多方控盤", "空方控盤"),
                        "RSI位階": (curr['RSI'] > 50, "強勢", "弱勢"),
                        "布林位階": (curr_p > curr['MA20'], "中軌上方", "中軌下方"),
                        "量能表現": (curr['Volume'] > curr['VMA20'], "放量", "縮量")
                    }
                    match_count = sum(1 for c, (cond, p, n) in conds.items() if cond)
                    score = int((match_count / len(conds)) * 100)
                    
                    # 2. 顯示得分與建議買點
                    st.progress(score / 100, text=f"📊 綜合診斷強度：{score}%")
                    
                    c1, c2 = st.columns(2)
                    c1.metric("📌 建議買點 (月線)", f"{entry_p:.2f}")
                    c2.metric("💰 目前現價", f"{curr_p:.2f}", delta=f"{curr_p - entry_p:.2f}")
                    
                    c3, c4 = st.columns(2)
                    c3.metric("🚫 止損價位", f"{sl_p:.2f}")
                    c4.metric("🎯 目標獲利", f"{tp_p:.2f}")
                    
                    st.divider()
                    
                    # --- 下方顯示詳細報告與圖表 ---
                    st.subheader("📋 指標詳細診斷")
                    d_cols = st.columns(3)
                    for i, (name, (cond, p, n)) in enumerate(conds.items()):
                        icon = "✅" if cond else "❌"
                        msg = p if cond else n
                        d_cols[i % 3].write(f"{icon} **{name}**: {msg}")
                    
                    st.subheader("📈 技術分析走勢")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    df_p = df.tail(60)
                    ax.plot(df_p.index, df_p['Close'], label='Price', color='#1c2833', lw=2)
                    ax.plot(df_p['MA20'], label='MA20 (買點參考)', color='#f1c40f', ls='--')
                    ax.fill_between(df_p.index, df_p['BB_up'], df_p['BB_low'], color='gray', alpha=0.1)
                    ax.axhline(sl_p, color='red', ls=':', alpha=0.3, label='Stop Loss')
                    ax.set_title(f"{query} ({sid}) 技術走勢")
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.warning("數據長度不足以計算指標 (需至少20日數據)")
            else:
                st.error(f"無法獲取 '{query}' 的數據，請檢查代碼。")
else:
    st.info("請在左側選單輸入股票代碼開始分析")
