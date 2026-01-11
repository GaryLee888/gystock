import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import twstock
import warnings

# --- 基礎設定 ---
st.set_page_config(page_title="台股全方位決策系統", layout="wide")
warnings.filterwarnings("ignore")

class StockMaster:
    def __init__(self):
        # 繼承原本的特殊映射
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
        # --- 核心指標 ---
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

        # --- 進階指標 ---
        mode = "Full" if data_len >= 60 else "Lite"
        if mode == "Full":
            df['ROC'] = df['Close'].pct_change(12) * 100
            df['MFI'] = 50 + (df['Close'].diff().rolling(14).mean() * 10)
            up_vol = df['Volume'].where(df['Close'] > df['Close'].shift(1), 0).rolling(10).sum()
            down_vol = df['Volume'].where(df['Close'] < df['Close'].shift(1), 0).rolling(10).sum()
            df['Vol_Ratio'] = up_vol / down_vol.replace(0, 1)
            df['SR_Rank'] = (df['Close'] - df['Close'].rolling(60).min()) / (df['Close'].rolling(60).max() - df['Close'].rolling(60).min()).replace(0, 1)
        
        return df.dropna(), mode

# --- 側邊欄 ---
with st.sidebar:
    st.title("🛡️ 交易參數設定")
    atr_mult = st.slider("ATR 止損倍數", 1.5, 3.5, 2.2)
    reward_ratio = st.slider("盈虧比 (TP)", 1.0, 5.0, 2.0)
    st.divider()
    st.header("🔍 批次名單")
    default_vals = ["2330", "2317", "2454", "能率亞洲", "2603", "2881", "3529", "8255", "", ""]
    input_queries = [st.text_input(f"股票 {i+1}", v, key=f"in_{i}") for i, v in enumerate(default_vals)]
    input_queries = [q for q in input_queries if q]

# --- 主畫面 ---
st.title("🚀 台股全方位決策系統")

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
                prev = df.iloc[-2]
                curr_p = float(curr['Close'])
                entry_p = float(curr['MA20'])
                sl_p = entry_p - (float(curr['ATR']) * atr_mult)
                tp_p = entry_p + (entry_p - sl_p) * reward_ratio

                # --- 20 項指標診斷邏輯 ---
                conds = {
                    "均線趨勢": (curr_p > curr['MA20'], "多頭", "空頭"),
                    "KD動能": (curr['K'] > curr['D'], "向上", "向下"),
                    "MACD柱": (curr['MACD_hist'] > 0, "紅柱", "綠柱"),
                    "RSI強弱": (curr['RSI'] > 50, "強勢", "弱勢"),
                    "布林位置": (curr_p > curr['MA20'], "中軌上", "中軌下"),
                    "短期排列": (curr['MA5'] > curr['MA10'], "向上", "糾結"),
                    "乖離安全": (abs(curr['BIAS20']) < 10, "安全", "過大"),
                    "量能狀態": (curr['Volume'] > curr['VMA20'], "放大", "萎縮"),
                    "短線力道": (curr_p > curr['MA5'], "強勁", "轉弱"),
                    "OBV籌碼": (curr['OBV'] >= df['OBV'].mean(), "集中", "渙散"),
                    "波動擠壓": (curr['BB_width'] < 0.1, "低波", "正常"),
                    "價格位階": (curr_p > curr['MA10'], "穩健", "偏弱"),
                    "動能加速": (curr['BIAS5'] > curr['BIAS20'], "加速", "趨緩")
                }
                
                if mode == "Full":
                    conds.update({
                        "價格變動率": (curr['ROC'] > 0, "正向", "負向"),
                        "資金流向": (curr['MFI'] > 50, "流入", "流出"),
                        "多空量比": (curr['Vol_Ratio'] > 1, "買盤強", "賣壓大"),
                        "60日位階": (curr['SR_Rank'] > 0.5, "健康", "偏低"),
                        "量價配合": (curr_p >= prev['Close'], "穩健", "背離"),
                        "支撐力道": (curr_p > curr['MA20'] * 0.98, "有撐", "破位"),
                        "長線保護": (curr['MA20'] > df['MA20'].shift(5).iloc[-1], "月線翻揚", "月線下彎")
                    })

                match_count = sum(1 for c, (cond, p, n) in conds.items() if cond)
                score = int((match_count / len(conds)) * 100)
                
                # --- 評分決策分級 ---
                if score <= 20: advice, color = "🚫 不能碰", "#7f8c8d"
                elif score <= 40: advice, color = "👀 看就好", "#95a5a6"
                elif score <= 60: advice, color = "⚖️ 中立觀望", "#3498db"
                elif score <= 80: advice, color = "💸 小量試單", "#f39c12"
                else: advice, color = "🔥 強烈買進", "#e74c3c"

                # --- 介面展示 ---
                st.markdown(f"<h2 style='color:{color}; text-align:center;'>{advice} (得分: {score})</h2>", unsafe_allow_html=True)
                st.progress(score / 100)
                
                st.divider()
                
                # 價位資訊卡片
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("💰 目前現價", f"{curr_p:.2f}")
                c2.metric("📌 建議買點", f"{entry_p:.2f}")
                c3.metric("🚫 停損點", f"{sl_p:.2f}")
                c4.metric("🎯 停利點", f"{tp_p:.2f}")
                
                # --- 方案一：Streamlit 原生互動圖表 ---
                st.subheader("📈 技術走勢圖 (手機互動版)")
                df_p = df.tail(60).copy()
                
                # 準備繪圖數據
                chart_data = df_p[['Close', 'MA20']].copy()
                chart_data.columns = ['現價/收盤價', '建議買點(月線)']
                
                # 繪製圖表
                st.line_chart(chart_data)
                st.caption(f"📊 輔助資訊：🔴 停損價 {sl_p:.2f} | 🟢 停利價 {tp_p:.2f} | 灰色區間為布林通道軌道")
                
                # 診斷細節
                with st.expander(f"🔍 查看完整 {len(conds)} 項診斷細節 ({mode} 模式)", expanded=False):
                    d_cols = st.columns(2)
                    for i, (name, (cond, p, n)) in enumerate(conds.items()):
                        d_cols[i % 2].write(f"{'🟢' if cond else '🔴'} **{name}**: {p if cond else n}")

            else:
                st.error(f"⚠️ {query} 數據不足，無法啟動分析。")
