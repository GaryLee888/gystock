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
        data_len = len(df)
        if data_len < 20: return None, "Insufficient"
        
        df = df.copy()
        # 核心指標 (20日即可計算)
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        df['BB_up'] = df['MA20'] + (std * 2)
        df['BB_low'] = df['MA20'] - (std * 2)
        df['BB_width'] = (df['BB_up'] - df['BB_low']) / df['MA20']
        
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
        df['BIAS5'] = (df['Close'] - df['MA5']) / df['MA5'] * 100
        df['BIAS20'] = (df['Close'] - df['MA20']) / df['MA20'] * 100
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

        # 進階指標 (需 60 日)
        mode = "Full" if data_len >= 60 else "Lite"
        if mode == "Full":
            df['ROC'] = df['Close'].pct_change(12) * 100
            df['MFI'] = 50 + (df['Close'].diff().rolling(14).mean() * 10)
            up_vol = df['Volume'].where(df['Close'] > df['Close'].shift(1), 0).rolling(10).sum()
            down_vol = df['Volume'].where(df['Close'] < df['Close'].shift(1), 0).rolling(10).sum()
            df['Vol_Ratio'] = up_vol / down_vol.replace(0, 1)
            df['SR_Rank'] = (df['Close'] - df['Close'].rolling(60).min()) / (df['Close'].rolling(60).max() - df['Close'].rolling(60).min()).replace(0, 1)
        
        return df.dropna(), mode

# --- UI 介面 ---
with st.sidebar:
    st.title("🛡️ 交易參數")
    atr_mult = st.slider("ATR 止損倍數", 1.5, 3.5, 2.2)
    reward_ratio = st.slider("盈虧比 (TP)", 1.0, 5.0, 2.0)
    st.divider()
    st.header("🔍 批次名單")
    default_vals = ["2330", "2317", "能率亞洲", "7861", "", "", "", "", "", ""]
    input_queries = [st.text_input(f"股票 {i+1}", v, key=f"in_{i}") for i, v in enumerate(default_vals)]
    input_queries = [q for q in input_queries if q]

st.title("🚀 台股多軌分析系統")

if input_queries:
    master = StockMaster()
    tabs = st.tabs([f"📊 {q}" for q in input_queries])
    
    for tab, query in zip(tabs, input_queries):
        with tab:
            sid = master.special_mapping.get(query, query)
            if not sid.isdigit():
                for code, info in twstock.codes.items():
                    if query in info.name: sid = code; break
            
            df_raw, _ = master.fetch_data(sid)
            df, mode = master.calculate_indicators(df_raw) if df_raw is not None else (None, None)
            
            if df is not None:
                curr = df.iloc[-1]
                curr_p = float(curr['Close'])
                entry_p = float(curr['MA20'])
                sl_p = entry_p - (float(curr['ATR']) * atr_mult)
                tp_p = entry_p + (entry_p - sl_p) * reward_ratio

                # 診斷逻辑分配
                conds = {
                    "均線趨勢": (curr_p > curr['MA20'], "多頭", "空頭"),
                    "KD動能": (curr['K'] > curr['D'], "向上", "向下"),
                    "MACD柱": (curr['MACD_hist'] > 0, "紅柱", "綠柱"),
                    "RSI強弱": (curr['RSI'] > 50, "強勢", "弱勢"),
                    "布林位置": (curr_p > curr['MA20'], "上位", "下位"),
                    "短期排列": (curr['MA5'] > curr['MA10'], "向上", "糾結"),
                    "乖離控制": (abs(curr['BIAS20']) < 10, "安全", "過大"),
                    "量能狀態": (curr['Volume'] > curr['VMA20'], "放大", "萎縮"),
                    "短線力道": (curr_p > curr['MA5'], "強勁", "轉弱"),
                    "OBV籌碼": (curr['OBV'] >= df['OBV'].mean(), "集中", "渙散")
                }
                
                if mode == "Full":
                    conds.update({
                        "價格變動": (curr['ROC'] > 0, "正向", "負向"),
                        "資金流向": (curr['MFI'] > 50, "流入", "流出"),
                        "買盤力道": (curr['Vol_Ratio'] > 1, "積極", "保守"),
                        "位階健康": (curr['SR_Rank'] > 0.5, "適中", "偏低"),
                        "動能加速": (curr['BIAS5'] > curr['BIAS20'], "加速", "趨緩")
                        # 此處可繼續增加至 20 項...
                    })

                match_count = sum(1 for c, (cond, p, n) in conds.items() if cond)
                score = int((match_count / len(conds)) * 100)
                
                # --- 頂部顯示 ---
                st.progress(score / 100, text=f"📊 [{mode} 模式] 診斷得分：{score}% ({match_count}/{len(conds)})")
                
                c1, c2 = st.columns(2)
                c1.metric("📌 建議買點", f"{entry_p:.2f}")
                c2.metric("💰 目前現價", f"{curr_p:.2f}", delta=f"{curr_p - entry_p:.2f}")
                
                # --- 詳細資訊 ---
                with st.expander(f"🔍 查看 {len(conds)} 項診斷清單", expanded=False):
                    d_cols = st.columns(2)
                    for i, (name, (cond, p, n)) in enumerate(conds.items()):
                        d_cols[i % 2].write(f"{'🟢' if cond else '🔴'} **{name}**: {p if cond else n}")

                st.subheader("📈 趨勢圖表")
                fig, ax = plt.subplots(figsize=(10, 4))
                df_p = df.tail(60)
                ax.plot(df_p.index, df_p['Close'], color='#1c2833', lw=2)
                ax.plot(df_p['MA20'], color='#f1c40f', ls='--')
                ax.fill_between(df_p.index, df_p['BB_up'], df_p['BB_low'], color='gray', alpha=0.1)
                st.pyplot(fig)
            else:
                st.error(f"⚠️ {query} 數據不足 (需至少 20 筆才能啟動精簡分析)")
