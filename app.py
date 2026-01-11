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
        if len(df) < 60: return None
        df = df.copy()
        
        # 1-3. 均線與排列
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        
        # 4-5. 布林軌道與寬度
        std = df['Close'].rolling(20).std()
        df['BB_up'] = df['MA20'] + (std * 2)
        df['BB_low'] = df['MA20'] - (std * 2)
        df['BB_width'] = (df['BB_up'] - df['BB_low']) / df['MA20']
        
        # 6-7. KD動能
        low_9 = df['Low'].rolling(9).min()
        high_9 = df['High'].rolling(9).max()
        df['K'] = ((df['Close'] - low_9) / (high_9 - low_9).replace(0, np.nan) * 100).ewm(com=2).mean()
        df['D'] = df['K'].ewm(com=2).mean()
        
        # 8. MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['MACD_hist'] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()
        
        # 9. RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss).replace(0, np.nan)))
        
        # 10. ATR (用於止損)
        tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        
        # 11-15. 能量與乖離
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['BIAS5'] = (df['Close'] - df['MA5']) / df['MA5'] * 100
        df['BIAS20'] = (df['Close'] - df['MA20']) / df['MA20'] * 100
        df['VMA20'] = df['Volume'].rolling(20).mean()
        df['ROC'] = df['Close'].pct_change(12) * 100
        
        # 16-20. 資金流與位階 (簡化算法移植)
        df['MFI'] = 50 + (df['Close'].diff().rolling(14).mean() * 10)
        up_vol = df['Volume'].where(df['Close'] > df['Close'].shift(1), 0).rolling(10).sum()
        down_vol = df['Volume'].where(df['Close'] < df['Close'].shift(1), 0).rolling(10).sum()
        df['Vol_Ratio'] = up_vol / down_vol.replace(0, 1)
        df['SR_Rank'] = (df['Close'] - df['Close'].rolling(60).min()) / (df['Close'].rolling(60).max() - df['Close'].rolling(60).min()).replace(0, 1)
        
        return df.dropna()

# --- 側邊欄 ---
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

# --- 主畫面 ---
st.title("🚀 台股全方位決策系統 (20指標版)")

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
                    
                    # 買賣點計算
                    entry_p = float(curr['MA20'])
                    sl_p = entry_p - (float(curr['ATR']) * atr_mult)
                    tp_p = entry_p + (entry_p - sl_p) * reward_ratio
                    
                    # --- 核心邏輯：20項診斷定義 ---
                    conds = {
                        "均線趨勢": (curr_p > curr['MA20'], "多頭趨勢", "空頭趨勢"),
                        "布林軌道": (curr_p > curr['MA20'], "軌道上位", "軌道下位"),
                        "KD動能": (curr['K'] > curr['D'], "動能向上", "動能向下"),
                        "MACD柱狀": (curr['MACD_hist'] > 0, "多方紅柱", "空方綠柱"),
                        "RSI強弱": (curr['RSI'] > 50, "強勢區間", "弱勢區間"),
                        "短期均線": (curr['MA5'] > curr['MA10'], "短期向上", "短期糾結"),
                        "威廉指標": (curr['K'] > 50, "多方主導", "空方主導"),
                        "乖離安全": (abs(curr['BIAS20']) < 10, "乖離正常", "乖離過大"),
                        "波動擠壓": (curr['BB_width'] < 0.1, "低波擠壓", "波幅正常"),
                        "量價配合": (curr_p >= prev['Close'], "量價穩健", "量價背離"),
                        "相對強度": (curr_p > curr['MA5'], "強於均值", "弱於均值"),
                        "OBV能量": (curr['OBV'] >= df['OBV'].mean(), "籌碼集中", "籌碼渙散"),
                        "資金流向": (curr['MFI'] > 50, "資金流入", "資金流出"),
                        "量能放大": (curr['Volume'] > curr['VMA20'], "量能放大", "量能萎縮"),
                        "短線強勁": (curr_p > curr['MA5'], "多方強勢", "多方轉弱"),
                        "乖離動能": (curr['BIAS5'] > curr['BIAS20'], "動能加速", "動能趨緩"),
                        "站穩支撐": (curr_p > curr['MA20'], "支撐強勁", "支撐轉弱"),
                        "買盤積極": (curr['Vol_Ratio'] > 1, "買盤主導", "賣壓主導"),
                        "價格變動": (curr['ROC'] > 0, "趨勢正向", "趨勢負向"),
                        "位階健康": (curr['SR_Rank'] > 0.5, "位階適中", "位階偏低")
                    }
                    
                    match_count = sum(1 for c, (cond, p, n) in conds.items() if cond)
                    score = int((match_count / 20) * 100)
                    
                    # --- 1. 最上方：分數與買點 ---
                    st.progress(score / 100, text=f"📊 綜合診斷強度：{score}% ({match_count}/20 指標符合)")
                    
                    c1, c2 = st.columns(2)
                    c1.metric("📌 建議買點 (月線)", f"{entry_p:.2f}")
                    c2.metric("💰 目前現價", f"{curr_p:.2f}", delta=f"{curr_p - entry_p:.2f}")
                    
                    c3, c4 = st.columns(2)
                    c3.metric("🚫 止損價位", f"{sl_p:.2f}")
                    c4.metric("🎯 目標獲利", f"{tp_p:.2f}")
                    
                    st.divider()

                    # --- 2. 中間：20項診斷清單 ---
                    with st.expander("🔍 查看完整 20 項診斷細節", expanded=False):
                        d_cols = st.columns(2)
                        items = list(conds.items())
                        for i in range(20):
                            name, (cond, p, n) = items[i]
                            icon = "🟢" if cond else "🔴"
                            msg = p if cond else n
                            d_cols[i % 2].write(f"{icon} **{name}**: {msg}")

                    # --- 3. 下方：技術分析圖表 ---
                    st.subheader("📈 技術分析走勢")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    df_p = df.tail(60)
                    ax.plot(df_p.index, df_p['Close'], label='收盤價', color='#1c2833', lw=2)
                    ax.plot(df_p['MA20'], label='MA20 (買點)', color='#f1c40f', ls='--')
                    ax.fill_between(df_p.index, df_p['BB_up'], df_p['BB_low'], color='gray', alpha=0.1)
                    ax.axhline(sl_p, color='red', ls=':', alpha=0.4, label='止損位')
                    ax.set_title(f"{query} 技術分析圖")
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.warning("歷史數據不足 (需至少 60 筆數據以計算位階指標)")
            else:
                st.error(f"無法獲取 '{query}' 數據")
