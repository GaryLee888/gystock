import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import twstock
import warnings

# --- 基礎設定 ---
st.set_page_config(page_title="台股極簡決策版", layout="wide")
warnings.filterwarnings("ignore")

# 注入 CSS 讓文字更小、間距更緊湊，適合手機一頁看完
st.markdown("""
    <style>
    .reportview-container .main .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .stMetric { padding: 0px 5px !important; }
    div[data-testid="stMarkdownContainer"] p { font-size: 13px !important; margin-bottom: 0px !important; }
    .stProgress > div > div > div > div { height: 10px !important; }
    </style>
    """, unsafe_allow_html=True)

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
    st.header("⚙️ 參數")
    atr_mult = st.number_input("ATR倍數", 1.0, 5.0, 2.2, 0.1)
    reward_ratio = st.number_input("盈虧比", 1.0, 5.0, 2.0, 0.1)
    queries = [st.text_input(f"股{i+1}", v, key=f"q{i}") for i, v in enumerate(["2330","2317","2454","能率亞洲","2603","2881"])]

# --- 主畫面 ---
if any(queries):
    master = StockMaster()
    tabs = st.tabs([f"{q}" for q in queries if q])
    
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
                    
                    # 診斷
                    conds = {
                        "MA": (curr_p > curr['MA20'], "多", "空"),
                        "BB": (curr_p > curr['MA20'], "上", "下"),
                        "KD": (curr['K'] > curr['D'], "↑", "↓"),
                        "MACD": (curr['MACD_hist'] > 0, "紅", "綠"),
                        "RSI": (curr['RSI'] > 50, "強", "弱"),
                        "排列": (curr['MA5'] > curr['MA10'], "正", "偏"),
                        "威廉": (curr['K'] > 50, "多", "空"),
                        "乖離": (abs(curr['BIAS20']) < 10, "安", "偏"),
                        "擠壓": (curr['BB_width'] < 0.1, "縮", "常"),
                        "量價": (curr_p >= prev['Close'], "穩", "背"),
                        "相對": (curr_p > prev['Close'], "優", "劣"),
                        "OBV": (curr['OBV'] >= df['OBV'].mean(), "集", "渙"),
                        "資金": (curr['MFI'] > 50, "入", "出"),
                        "均量": (curr['Volume'] > curr['VMA20'], "增", "縮"),
                        "短勁": (curr_p > curr['MA5'], "強", "弱"),
                        "加速": (curr['BIAS5'] > curr['BIAS20'], "加", "減"),
                        "支撐": (curr_p > curr['MA20'], "穩", "沉"),
                        "量比": (curr['Vol_Ratio'] > 1, "積極", "賣壓"),
                        "趨勢": (curr['ROC'] > 0, "正", "負"),
                        "位階": (curr_p > df['Close'].tail(60).min(), "健", "低")
                    }
                    
                    match_count = sum(1 for k, (cond, p, n) in conds.items() if cond)
                    score = int((match_count / 20) * 100)
                    
                    # 決策邏輯
                    if score <= 20: advice, color = "不能碰", "grey"
                    elif score <= 40: advice, color = "建議觀望", "orange"
                    elif score <= 60: advice, color = "中立", "blue"
                    elif score <= 80: advice, color = "小量試單", "green"
                    else: advice, color = "強烈買進", "red"

                    # --- 第一層：評分與建議 ---
                    st.write(f"### **{score}分 | :{color}[{advice}]**")
                    st.progress(score/100)
                    
                    # --- 第二層：核心價位 ---
                    entry_p = float(curr['MA20'])
                    sl_p = entry_p - (float(curr['ATR']) * atr_mult)
                    tp_p = entry_p + (entry_p - sl_p) * reward_ratio
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("現價", f"{curr_p:.0f}")
                    c2.metric("買點", f"{entry_p:.0f}")
                    c3.metric("止損", f"{sl_p:.0f}")
                    c4.metric("獲利", f"{tp_p:.0f}")

                    # --- 第三層：20項診斷網格 (5x4) ---
                    st.write("---")
                    items = list(conds.items())
                    rows = [items[i:i + 4] for i in range(0, len(items), 4)]
                    
                    for row in rows:
                        cols = st.columns(4)
                        for i, (name, (cond, p, n)) in enumerate(row):
                            icon = "●" # 簡化圓點
                            clr = "green" if cond else "red"
                            cols[i].markdown(f":{clr}[{icon}] **{name}**\n\n{p if cond else n}")
                else: st.error("數據不足")
            else: st.error("無數據")
