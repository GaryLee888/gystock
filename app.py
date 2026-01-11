import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import twstock
import warnings

# --- 基礎設定 ---
st.set_page_config(page_title="台股全方位分析", layout="wide")
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
        df['BB_width'] = (df['BB_up'] - df['BB_low']) / df['MA20']
        low_9, high_9 = df['Low'].rolling(9).min(), df['High'].rolling(9).max()
        df['K'] = ((df['Close'] - low_9) / (high_9 - low_9).replace(0, np.nan) * 100).ewm(com=2).mean()
        df['D'] = df['K'].ewm(com=2).mean()
        ema12, ema26 = df['Close'].ewm(span=12).mean(), df['Close'].ewm(span=26).mean()
        df['MACD_hist'] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss).replace(0, np.nan)))
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['MFI'] = 50 + (df['Close'].diff().rolling(14).mean() * 10)
        tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        df['VMA20'] = df['Volume'].rolling(20).mean()
        df['BIAS5'] = (df['Close'] - df['MA5']) / df['MA5'] * 100
        df['BIAS20'] = (df['Close'] - df['MA20']) / df['MA20'] * 100
        df['ROC'] = df['Close'].pct_change(12) * 100
        up_vol = df['Volume'].where(df['Close'] > df['Close'].shift(1), 0).rolling(10).sum()
        down_vol = df['Volume'].where(df['Close'] < df['Close'].shift(1), 0).rolling(10).sum()
        df['Vol_Ratio'] = up_vol / down_vol.replace(0, 1)
        return df.dropna()

# --- 側邊欄 ---
with st.sidebar:
    st.title("⚙️ 參數設定")
    atr_mult = st.number_input("ATR 止損倍數", 1.0, 5.0, 2.2)
    reward_ratio = st.number_input("盈虧比", 1.0, 5.0, 2.0)
    st.divider()
    default_stocks = ["2330", "2317", "2454", "能率亞洲", "2603", "2881", "", "", "", ""]
    queries = [st.text_input(f"股票 {i+1}", v, key=f"q{i}") for i, v in enumerate(default_stocks) if v or i < 6]

# --- 主畫面 ---
st.title("💹 台股全方位決策系統")

if any(queries):
    master = StockMaster()
    tabs = st.tabs([f"🔍 {q}" for q in queries if q])
    
    for tab, query in zip(tabs, [q for q in queries if q]):
        with tab:
            sid = master.special_mapping.get(query, query)
            if not sid.isdigit():
                for code, info in twstock.codes.items():
                    if query in info.name: sid = code; break
            
            df_raw, _ = master.fetch_data(sid)
            if df_raw is not None:
                df = master.calculate_indicators(df_raw)
                if df is not None:
                    curr, prev = df.iloc[-1], df.iloc[-2]
                    curr_p = float(curr['Close'])
                    
                    # 診斷邏輯
                    conds = {
                        "均線趨勢": (curr_p > curr['MA20'], "多頭", "空頭"),
                        "布林軌道": (curr_p > curr['MA20'], "上位", "下位"),
                        "KD動能": (curr['K'] > curr['D'], "向上", "向下"),
                        "MACD趨勢": (curr['MACD_hist'] > 0, "多方", "空方"),
                        "RSI強弱": (curr['RSI'] > 50, "強勢", "弱勢"),
                        "多頭排列": (curr['MA5'] > curr['MA10'], "向上", "糾結"),
                        "威廉指標": (curr['K'] > 50, "多主", "空主"),
                        "乖離控制": (abs(curr['BIAS20']) < 10, "安全", "偏離"),
                        "低波擠壓": (curr['BB_width'] < 0.1, "擠壓", "正常"),
                        "量價配合": (curr_p >= prev['Close'], "穩健", "背離"),
                        "相對強度": (curr_p > prev['Close'], "優勢", "弱勢"),
                        "籌碼OBV": (curr['OBV'] >= df['OBV'].mean(), "集中", "渙散"),
                        "資金流向": (curr['MFI'] > 50, "流入", "流出"),
                        "成交均量": (curr['Volume'] > curr['VMA20'], "放大", "萎縮"),
                        "短線勁道": (curr_p > curr['MA5'], "強勁", "轉弱"),
                        "動能加速": (curr['BIAS5'] > curr['BIAS20'], "加速", "趨緩"),
                        "站穩支撐": (curr_p > curr['MA20'], "穩固", "沉重"),
                        "多空量比": (curr['Vol_Ratio'] > 1, "積極", "較大"),
                        "趨勢變動": (curr['ROC'] > 0, "正向", "負向"),
                        "位階評估": (curr_p > df['Close'].tail(60).min(), "健康", "偏低")
                    }
                    
                    match_count = sum(1 for k, (cond, p, n) in conds.items() if cond)
                    score = int((match_count / 20) * 100)
                    
                    # --- 置頂決策區 ---
                    res_col1, res_col2 = st.columns([1, 1])
                    with res_col1:
                        st.metric("核心評分", f"{score} 分")
                    with res_col2:
                        advice = "🚀 建議買進" if score >= 70 else "⚖️ 建議觀望" if score >= 50 else "⚠️ 避開標的"
                        color = "green" if score >= 70 else "orange" if score >= 50 else "red"
                        st.markdown(f"### 決策：:{color}[{advice}]")
                    
                    st.progress(score/100)
                    st.divider()

                    # --- 數據卡片 ---
                    entry_p = float(curr['MA20'])
                    sl_p = entry_p - (float(curr['ATR']) * atr_mult)
                    tp_p = entry_p + (entry_p - sl_p) * reward_ratio
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("現價", f"{curr_p:.2f}")
                    c2.metric("買點", f"{entry_p:.1f}")
                    c3.metric("止損", f"{sl_p:.1f}")
                    c4.metric("獲利", f"{tp_p:.1f}")

                    # --- 20項指標 10x2 排版 ---
                    st.subheader("📊 20項綜合診斷 (10個一列)")
                    items = list(conds.items())
                    col_a, col_b = st.columns(2)
                    
                    with col_a: # 前 10 個
                        for i in range(10):
                            name, (cond, p, n) = items[i]
                            icon = "🟢" if cond else "🔴"
                            st.write(f"{icon} {name}: **{p if cond else n}**")
                            
                    with col_b: # 後 10 個
                        for i in range(10, 20):
                            name, (cond, p, n) = items[i]
                            icon = "🟢" if cond else "🔴"
                            st.write(f"{icon} {name}: **{p if cond else n}**")
                    
                    # --- 圖表 ---
                    st.divider()
                    fig, ax = plt.subplots(figsize=(10, 4))
                    df_p = df.tail(65)
                    ax.plot(df_p.index, df_p['Close'], label='Price', color='#1c2833', lw=1.5)
                    ax.plot(df_p['MA20'], label='MA20', color='#f1c40f', ls='--')
                    ax.fill_between(df_p.index, df_p['BB_up'], df_p['BB_low'], alpha=0.1, color='gray')
                    ax.legend(prop={'size': 8})
                    st.pyplot(fig)
                else:
                    st.error("數據不足")
            else:
                st.error("查無數據")
