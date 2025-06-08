# ============================================
# STREAMLIT UI MELHORADA - STOCKAI PREDICTOR
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import joblib
import ta
from datetime import datetime, timedelta
import threading
import queue
import os
import time
import warnings
from sklearn.preprocessing import RobustScaler
import gc

# Suprimir warnings
warnings.filterwarnings('ignore')

# Configurar TensorFlow para evitar problemas de GPU
try:
    tf.config.experimental.set_visible_devices([], 'GPU')
except:
    pass

# Definir a classe DAINLayer ANTES de tudo - CORREÇÃO PRINCIPAL
@tf.keras.utils.register_keras_serializable()
class DAINLayer(Layer):
    """Deep Adaptive Input Normalization for non-stationary financial data"""
    def __init__(self, **kwargs):
        super(DAINLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.shift = self.add_weight(name='shift', 
                                    shape=(input_shape[-1],),
                                    initializer='zeros',
                                    trainable=True)
        self.scale = self.add_weight(name='scale',
                                    shape=(input_shape[-1],),
                                    initializer='ones',
                                    trainable=True)
        self.gate = self.add_weight(name='gate',
                                   shape=(input_shape[-1],),
                                   initializer='ones',
                                   trainable=True)
        super().build(input_shape)
    
    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=1, keepdims=True)
        std = tf.sqrt(variance + 1e-8)
        
        normalized = (inputs - mean) / std
        transformed = normalized * self.scale + self.shift
        gated = transformed * tf.nn.sigmoid(self.gate)
        
        return gated
    
    def get_config(self):
        config = super().get_config()
        return config

# Tentar importar google.generativeai
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning("⚠️ Google Generative AI não instalado. Instale com: pip install google-generativeai")

# Configuração da página
st.set_page_config(
    page_title="StockAI Predictor 📈",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado melhorado
st.markdown("""
<style>
    /* Tema moderno */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #2c3e50;
    }

    /* Cards com efeito glassmorphism */
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        color: #2c3e50;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Pequenos cards de métricas */
    .mini-metric-card {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.4);
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .mini-metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12);
    }

    /* Header com gradiente animado */
    .dashboard-header {
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Botões com efeito moderno */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        background: linear-gradient(45deg, #764ba2, #667eea);
    }

    /* Métricas destacadas com animação */
    .big-metric {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    /* Cards de insight melhorados */
    .insight-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
    }

    .insight-card:hover {
        transform: translateX(5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
    }

    /* Sidebar aprimorada */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }

    /* Status indicators */
    .status-success {
        color: #27ae60;
        font-weight: bold;
    }

    .status-warning {
        color: #f39c12;
        font-weight: bold;
    }

    .status-error {
        color: #e74c3c;
        font-weight: bold;
    }

    /* Loading animation melhorada */
    .loading-pulse {
        animation: loading-pulse 1.5s ease-in-out infinite;
    }

    @keyframes loading-pulse {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }

    /* Tabs estilizadas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.1);
        padding: 0.5rem;
        border-radius: 15px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    /* Alertas personalizados */
    .custom-alert {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid;
    }

    .alert-success {
        background-color: #d4edda;
        border-left-color: #28a745;
        color: #155724;
    }

    .alert-warning {
        background-color: #fff3cd;
        border-left-color: #ffc107;
        color: #856404;
    }

    .alert-error {
        background-color: #f8d7da;
        border-left-color: #dc3545;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Inicialização do estado da sessão
if 'insights_queue' not in st.session_state:
    st.session_state.insights_queue = queue.Queue()
if 'insights_data' not in st.session_state:
    st.session_state.insights_data = {}
if 'model_configured' not in st.session_state:
    st.session_state.model_configured = False
if 'last_ticker' not in st.session_state:
    st.session_state.last_ticker = None
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}
if 'loading_logs' not in st.session_state:
    st.session_state.loading_logs = []
if 'verbose_mode' not in st.session_state:
    st.session_state.verbose_mode = False

# Configurações dos índices de mercado (igual ao treinamento)
MARKET_INDICES = {
    '^BVSP': 'Ibovespa',
    '^DJI': 'Dow Jones',
    'CL=F': 'Petróleo WTI',
    'BRL=X': 'USD/BRL',
    '^VIX': 'VIX',
    'GC=F': 'Ouro'
}

# === FUNÇÕES AUXILIARES MELHORADAS ===

def extrair_valor_escalar(data):
    """Extrai valor escalar de qualquer tipo de dado pandas/numpy com verificações de segurança"""
    try:
        if data is None:
            return 0.0
        if isinstance(data, (int, float)):
            return float(data)
        if isinstance(data, np.ndarray):
            if len(data.flatten()) == 0:
                return 0.0
            return float(data.flatten()[0])
        if isinstance(data, pd.Series):
            if len(data) == 0:
                return 0.0
            # Verificar se há valores válidos
            valid_data = data.dropna()
            if len(valid_data) == 0:
                return 0.0
            return float(valid_data.iloc[0])
        if isinstance(data, pd.DataFrame):
            if data.empty:
                return 0.0
            # Verificar se há valores válidos
            valid_data = data.dropna()
            if valid_data.empty:
                return 0.0
            return float(valid_data.iloc[0, 0])
        return float(data)
    except (ValueError, TypeError, IndexError, KeyError):
        return 0.0

def log_verbose(message, log_type="info"):
    """Adiciona log apenas se modo verboso estiver ativo"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.loading_logs.append((log_type, log_entry))
    
    if st.session_state.verbose_mode:
        if log_type == "info":
            st.info(message)
        elif log_type == "success":
            st.success(message)
        elif log_type == "warning":
            st.warning(message)
        elif log_type == "error":
            st.error(message)

def verificar_saude_dados(dados):
    """Verifica a qualidade dos dados carregados"""
    if dados is None or dados.empty:
        return False, "Dados vazios ou não carregados"
    
    problemas = []
    
    # Verificar colunas essenciais
    cols_essenciais = ['Open', 'High', 'Low', 'Close', 'Volume']
    cols_faltantes = [col for col in cols_essenciais if col not in dados.columns]
    if cols_faltantes:
        problemas.append(f"Colunas faltantes: {cols_faltantes}")
    
    # Verificar quantidade de dados
    if len(dados) < 10:
        problemas.append(f"Muito poucos dados: {len(dados)} registros (mínimo: 10)")
    elif len(dados) < 30:
        problemas.append(f"Poucos dados para análise completa: {len(dados)} registros (recomendado: 30+)")
    
    # Verificar valores NaN excessivos
    if len(dados) > 0:
        nan_pct = dados.isnull().sum().sum() / (len(dados) * len(dados.columns)) * 100
        if nan_pct > 20:
            problemas.append(f"Muitos valores NaN: {nan_pct:.1f}%")
    
    # Verificar preços zerados
    if 'Close' in dados.columns and len(dados) > 0:
        zeros_pct = (dados['Close'] == 0).sum() / len(dados) * 100
        if zeros_pct > 5:
            problemas.append(f"Muitos preços zerados: {zeros_pct:.1f}%")
        
        # Verificar se há variação nos preços
        if dados['Close'].nunique() <= 1:
            problemas.append("Preços sem variação")
    
    if problemas:
        return False, "; ".join(problemas)
    
    return True, "Dados OK"

@st.cache_data(ttl=300)  # Cache por 5 minutos
def coletar_dados_mercado_expandido(inicio, fim):
    """Coleta dados expandidos dos índices de mercado (versão silenciosa)"""
    market_data = {}
    valid_data = []
    
    for i, (symbol, name) in enumerate(MARKET_INDICES.items()):
        try:
            data = yf.download(symbol, start=inicio, end=fim, progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
            
            if len(data) > 0 and 'Close' in data.columns:
                temp_df = pd.DataFrame(index=data.index)
                
                close_series = pd.Series(data['Close'].values, index=data.index)
                high_series = pd.Series(data.get('High', data['Close']).values, index=data.index)
                low_series = pd.Series(data.get('Low', data['Close']).values, index=data.index)
                volume_series = pd.Series(data.get('Volume', 0).values, index=data.index)
                
                # Múltiplas features para cada índice
                temp_df[f'{name}_Close'] = close_series
                temp_df[f'{name}_Volume'] = volume_series
                temp_df[f'{name}_Return'] = close_series.pct_change()
                temp_df[f'{name}_Log_Return'] = np.log(close_series / close_series.shift(1))
                temp_df[f'{name}_Volatility'] = temp_df[f'{name}_Return'].rolling(20).std()
                temp_df[f'{name}_SMA20'] = close_series.rolling(20).mean()
                
                try:
                    temp_df[f'{name}_RSI'] = ta.momentum.rsi(close_series, window=14)
                except:
                    temp_df[f'{name}_RSI'] = 50.0
                
                valid_data.append(temp_df)
                log_verbose(f"✅ Dados de {name} coletados", "success")
                
        except Exception as e:
            log_verbose(f"⚠️ Erro ao coletar {name}: {str(e)}", "warning")
    
    if valid_data:
        result_df = valid_data[0]
        for df in valid_data[1:]:
            result_df = result_df.join(df, how='outer')
        log_verbose(f"✅ Dados de mercado adicionados: {len(result_df.columns)} features", "success")
        return result_df
    else:
        log_verbose("⚠️ Nenhum dado de mercado coletado", "warning")
        return pd.DataFrame()

def aplicar_wavelet_denoising(signal, wavelet='db4', level=1):
    """Aplica wavelet denoising (versão simplificada)"""
    try:
        import pywt
        coeff = pywt.wavedec(signal, wavelet, mode="per")
        sigma = (1/0.6745) * np.median(np.abs(coeff[-level]))
        uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:])
        return pywt.waverec(coeff, wavelet, mode='per')[:len(signal)]
    except:
        return signal

def adicionar_indicadores_tecnicos_completos(df):
    """Adiciona conjunto completo de indicadores técnicos compatível com treinamento (versão silenciosa)"""
    if df is None or df.empty:
        return None
    
    try:
        df = df.copy()
        
        # Verificar colunas necessárias
        required_columns = ['Close', 'Volume', 'High', 'Low', 'Open']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            log_verbose(f"❌ Colunas obrigatórias ausentes: {missing_columns}", "error")
            return None
        
        # Extrair séries de forma robusta
        def extract_and_clean_series(column_data, fill_value=None):
            if isinstance(column_data, pd.DataFrame):
                series_data = column_data.iloc[:, 0]
            else:
                series_data = column_data
            series_data = series_data.squeeze()
            
            if fill_value is not None:
                series_data = series_data.fillna(fill_value)
            else:
                series_data = series_data.ffill().bfill()
            
            return series_data
        
        close = extract_and_clean_series(df['Close'])
        high = extract_and_clean_series(df['High'])
        low = extract_and_clean_series(df['Low'])
        volume = extract_and_clean_series(df['Volume'], 0)
        open_price = extract_and_clean_series(df['Open'])
        
        if len(close) < 30:
            log_verbose(f"❌ Dados insuficientes: {len(close)} registros. Mínimo: 30", "error")
            return None
        
        # Aplicar wavelet denoising
        close_denoised = aplicar_wavelet_denoising(close.values)
        high_denoised = aplicar_wavelet_denoising(high.values)
        low_denoised = aplicar_wavelet_denoising(low.values)
        
        close_series = pd.Series(close_denoised, index=close.index)
        high_series = pd.Series(high_denoised, index=high.index)
        low_series = pd.Series(low_denoised, index=low.index)
        volume_series = pd.Series(volume.values, index=volume.index)
        
        indicators_count = 0
        
        # === Price Transformations ===
        try:
            df['Log_Return'] = np.log(close_series / close_series.shift(1))
            df['Log_Return_2'] = np.log(close_series / close_series.shift(2))
            df['Log_Return_5'] = np.log(close_series / close_series.shift(5))
            df['Log_Return_10'] = np.log(close_series / close_series.shift(10))
            df['Log_Return_20'] = np.log(close_series / close_series.shift(20))
            indicators_count += 5
        except:
            pass
        
        # === Moving Averages ===
        for period in [5, 10, 15, 20, 30, 50, 100, 200]:
            try:
                df[f'SMA_{period}'] = ta.trend.sma_indicator(close_series, window=period)
                df[f'EMA_{period}'] = ta.trend.ema_indicator(close_series, window=period)
                df[f'Price_to_SMA_{period}'] = close_series / df[f'SMA_{period}']
                df[f'Price_to_EMA_{period}'] = close_series / df[f'EMA_{period}']
                indicators_count += 4
            except:
                continue
        
        # === Momentum Indicators ===
        for period in [7, 14, 21, 28]:
            try:
                df[f'RSI_{period}'] = ta.momentum.rsi(close_series, window=period)
                df[f'RSI_{period}_Signal'] = df[f'RSI_{period}'].rolling(3).mean()
                indicators_count += 2
            except:
                continue
        
        # MACD variations
        macd_configs = [(12, 26, 9), (5, 35, 5), (8, 17, 9)]
        for i, (fast, slow, signal) in enumerate(macd_configs):
            try:
                macd = ta.trend.MACD(close_series, window_slow=slow, window_fast=fast, window_sign=signal)
                df[f'MACD_{i+1}'] = macd.macd()
                df[f'MACD_Signal_{i+1}'] = macd.macd_signal()
                df[f'MACD_Diff_{i+1}'] = macd.macd_diff()
                indicators_count += 3
            except:
                continue
        
        # === Volatility Indicators ===
        for period in [10, 20, 30]:
            try:
                bb = ta.volatility.BollingerBands(close_series, window=period, window_dev=2)
                df[f'BB_Upper_{period}'] = bb.bollinger_hband()
                df[f'BB_Lower_{period}'] = bb.bollinger_lband()
                df[f'BB_Middle_{period}'] = bb.bollinger_mavg()
                df[f'BB_Width_{period}'] = bb.bollinger_wband()
                df[f'BB_Position_{period}'] = bb.bollinger_pband()
                indicators_count += 5
            except:
                continue
        
        for period in [7, 14, 21, 28]:
            try:
                df[f'ATR_{period}'] = ta.volatility.average_true_range(high_series, low_series, close_series, window=period)
                df[f'ATR_Ratio_{period}'] = df[f'ATR_{period}'] / close_series
                indicators_count += 2
            except:
                continue
        
        # === Volume Indicators ===
        try:
            df['OBV'] = ta.volume.on_balance_volume(close_series, volume_series)
            df['OBV_EMA'] = ta.trend.ema_indicator(df['OBV'], window=20)
            df['CMF'] = ta.volume.chaikin_money_flow(high_series, low_series, close_series, volume_series)
            df['FI'] = ta.volume.force_index(close_series, volume_series)
            indicators_count += 4
            
            for period in [7, 14, 21]:
                df[f'MFI_{period}'] = ta.volume.money_flow_index(high_series, low_series, close_series, volume_series, window=period)
                indicators_count += 1
            
            df['VWAP'] = ta.volume.volume_weighted_average_price(high_series, low_series, close_series, volume_series)
            df['Price_to_VWAP'] = close_series / df['VWAP']
            indicators_count += 2
        except:
            pass
        
        # === Trend Indicators ===
        for period in [7, 14, 21, 28]:
            try:
                adx = ta.trend.ADXIndicator(high_series, low_series, close_series, window=period)
                df[f'ADX_{period}'] = adx.adx()
                df[f'ADX_Pos_{period}'] = adx.adx_pos()
                df[f'ADX_Neg_{period}'] = adx.adx_neg()
                indicators_count += 3
            except:
                continue
        
        # === Time-based Features ===
        df['Day_of_Week'] = df.index.dayofweek
        df['Day_of_Month'] = df.index.day
        df['Week_of_Year'] = df.index.isocalendar().week
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['Is_Month_Start'] = (df.index.day <= 5).astype(int)
        df['Is_Month_End'] = (df.index.day >= 25).astype(int)
        df['Is_Quarter_End'] = ((df.index.month % 3 == 0) & (df.index.day >= 25)).astype(int)
        indicators_count += 8
        
        # === Statistical Features ===
        for window in [5, 10, 20, 30, 60]:
            try:
                df[f'Rolling_Mean_{window}'] = close_series.rolling(window).mean()
                df[f'Rolling_Std_{window}'] = close_series.rolling(window).std()
                df[f'Rolling_Skew_{window}'] = close_series.rolling(window).skew()
                df[f'Rolling_Kurt_{window}'] = close_series.rolling(window).kurt()
                df[f'Rolling_Min_{window}'] = close_series.rolling(window).min()
                df[f'Rolling_Max_{window}'] = close_series.rolling(window).max()
                indicators_count += 6
            except:
                continue
        
        # === Pattern Recognition ===
        df['Body_Size'] = abs(close_series - open_price) / close_series
        df['Upper_Shadow'] = (high_series - np.maximum(close_series, open_price)) / close_series
        df['Lower_Shadow'] = (np.minimum(close_series, open_price) - low_series) / close_series
        indicators_count += 3
        
        # === Crossover Signals ===
        try:
            df['Golden_Cross'] = ((df['SMA_50'] > df['SMA_200']) & 
                                 (df['SMA_50'].shift(1) <= df['SMA_200'].shift(1))).astype(int)
            df['Death_Cross'] = ((df['SMA_50'] < df['SMA_200']) & 
                                (df['SMA_50'].shift(1) >= df['SMA_200'].shift(1))).astype(int)
            indicators_count += 2
        except:
            df['Golden_Cross'] = 0
            df['Death_Cross'] = 0
        
        # Manter compatibilidade com UI antiga
        df['SMA_7'] = df.get('SMA_5', ta.trend.sma_indicator(close_series, window=7))
        df['SMA_21'] = df.get('SMA_20', ta.trend.sma_indicator(close_series, window=21))
        df['EMA_9'] = df.get('EMA_10', ta.trend.ema_indicator(close_series, window=9))
        df['RSI'] = df.get('RSI_14', ta.momentum.rsi(close_series, window=14))
        
        macd = ta.trend.MACD(close_series)
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        
        bb = ta.volatility.BollingerBands(close_series, window=20)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        
        # Volume indicators para compatibilidade
        volume_sma = ta.trend.sma_indicator(volume_series, window=10)
        df['Volume_ratio'] = volume_series / volume_sma.replace(0, np.nan)
        df['High_Low_pct'] = (high_series - low_series) / close_series * 100
        df['Price_change'] = close_series.pct_change()
        
        log_verbose(f"✅ {indicators_count}+ indicadores técnicos calculados", "success")
        return df
        
    except Exception as e:
        log_verbose(f"❌ Erro ao calcular indicadores técnicos: {str(e)}", "error")
        return None

@st.cache_data(ttl=300)
def carregar_dados_ticker(ticker, periodo='1y'):
    """Carrega dados do ticker com cache melhorado (versão silenciosa)"""
    try:
        # Limpar logs anteriores
        if not st.session_state.verbose_mode:
            st.session_state.loading_logs = []
        
        log_verbose(f"🔄 Carregando dados para {ticker} (período: {periodo})...")
        
        # Calcular datas
        fim = datetime.now()
        periodos_map = {
            '1mo': 30, '3mo': 90, '6mo': 180,
            '1y': 365, '2y': 730, '5y': 1825
        }
        dias = periodos_map.get(periodo, 365)
        inicio = fim - timedelta(days=dias)
        
        # Download dos dados principais
        dados = yf.download(
            ticker, 
            start=inicio,
            end=fim,
            progress=False,
            timeout=30,
            auto_adjust=False
        )
        
        if dados is None or dados.empty:
            log_verbose(f"❌ Nenhum dado encontrado para {ticker}", "error")
            return None
        
        # Verificar se obtivemos dados suficientes antes de continuar
        if len(dados) < 5:
            log_verbose(f"❌ Muito poucos dados baixados: {len(dados)} registros", "error")
            return None
        
        # Flatten columns
        if isinstance(dados.columns, pd.MultiIndex):
            dados.columns = [col[0] if isinstance(col, tuple) else col for col in dados.columns]
        
        # Verificar saúde dos dados
        saude_ok, msg_saude = verificar_saude_dados(dados)
        if not saude_ok:
            log_verbose(f"❌ Problemas nos dados: {msg_saude}", "error")
            return None
        
        log_verbose(f"✅ {len(dados)} registros baixados com sucesso", "success")
        
        # Coletar dados de mercado
        log_verbose("📈 Coletando dados de mercado correlacionados...", "info")
        dados_mercado = coletar_dados_mercado_expandido(inicio, fim)
        if len(dados_mercado) > 0:
            dados = dados.join(dados_mercado, how='left')
        else:
            log_verbose("⚠️ Não foi possível coletar dados de mercado", "warning")
        
        # Adicionar indicadores técnicos
        log_verbose("📊 Calculando indicadores técnicos avançados...", "info")
        dados_com_indicadores = adicionar_indicadores_tecnicos_completos(dados)
        
        if dados_com_indicadores is None:
            log_verbose("❌ Erro ao calcular indicadores técnicos", "error")
            return None
        
        # Limpeza final
        dados_com_indicadores.replace([np.inf, -np.inf], np.nan, inplace=True)
        dados_com_indicadores.ffill(inplace=True)
        dados_com_indicadores.dropna(inplace=True)
        
        # Verificação final de dados suficientes
        if len(dados_com_indicadores) < 10:
            log_verbose(f"❌ Dados insuficientes após processamento: {len(dados_com_indicadores)} registros", "error")
            return None
        
        # Verificar se colunas essenciais ainda existem
        if 'Close' not in dados_com_indicadores.columns or dados_com_indicadores['Close'].empty:
            log_verbose("❌ Coluna Close perdida durante processamento", "error")
            return None
        
        log_verbose(f"✅ Dados carregados: {len(dados_com_indicadores)} registros, {len(dados_com_indicadores.columns)} features", "success")
        return dados_com_indicadores
        
    except Exception as e:
        log_verbose(f"❌ Erro ao carregar dados: {str(e)}", "error")
        return None

def carregar_modelo_completo(ticker):
    """Carrega modelo com custom objects e cache"""
    try:
        # Verificar cache
        if ticker in st.session_state.model_cache:
            return st.session_state.model_cache[ticker]
        
        # Caminhos dos arquivos
        model_path = f'models/{ticker}_directional_model.keras'
        scaler_path = f'scalers/{ticker}_directional_scaler.pkl'
        metrics_path = f'metrics/{ticker}_directional_metrics.pkl'
        
        # Verificar existência
        if not all(os.path.exists(path) for path in [model_path, scaler_path, metrics_path]):
            missing = [path for path in [model_path, scaler_path, metrics_path] if not os.path.exists(path)]
            return None, None, None, f"Arquivos faltantes: {missing}"
        
        # Carregar métricas
        metricas = joblib.load(metrics_path)
        
        # Carregar scaler
        scaler = joblib.load(scaler_path)
        
        # Carregar modelo com custom objects
        custom_objects = {'DAINLayer': DAINLayer}
        modelo = load_model(model_path, custom_objects=custom_objects, compile=False)
        
        # Cache
        st.session_state.model_cache[ticker] = (modelo, scaler, metricas, None)
        
        return modelo, scaler, metricas, None
        
    except Exception as e:
        error_msg = f"Erro ao carregar modelo: {str(e)}"
        return None, None, None, error_msg

def preparar_dados_previsao(dados, metricas):
    """Prepara dados para previsão seguindo exatamente o padrão de treinamento"""
    try:
        # Features são todas as colunas EXCETO preços básicos
        feature_cols = [col for col in dados.columns 
                       if col not in ['Close', 'Open', 'High', 'Low', 'Adj Close', 'Volume']]
        
        # Usar features selecionadas se disponível
        if 'feature_names' in metricas:
            feature_names = metricas['feature_names']
            
            # Preencher features faltantes com 0
            for feature in feature_names:
                if feature not in dados.columns:
                    dados[feature] = 0.0
            
            # Selecionar e ordenar features
            dados_features = dados[feature_names].copy()
        else:
            dados_features = dados[feature_cols].copy()
        
        # Limpeza
        dados_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        dados_features.ffill(inplace=True)
        dados_features.fillna(0, inplace=True)
        
        return dados_features.values
        
    except Exception as e:
        st.error(f"❌ Erro ao preparar dados: {str(e)}")
        return None

# === FUNÇÕES DE IA GENERATIVA ===

def configurar_gemini(api_key):
    """Configura o modelo Gemini"""
    if not GEMINI_AVAILABLE:
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        return model
    except Exception as e:
        st.error(f"Erro ao configurar Gemini: {str(e)}")
        return None

def gerar_insight_async(model, prompt, insight_type):
    """Gera insight de forma assíncrona"""
    if not GEMINI_AVAILABLE:
        st.session_state.insights_queue.put((insight_type, "Google Generative AI não disponível"))
        return
    
    try:
        safety_settings = [
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"}
        ]

        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=500,
            )
        )
        st.session_state.insights_queue.put((insight_type, response.text))
    except Exception as e:
        st.session_state.insights_queue.put((insight_type, f"Erro ao gerar insight: {str(e)}"))

def iniciar_geracao_insights(model, ticker, dados, metricas):
    """Inicia geração de insights com dados extraídos corretamente"""
    try:
        # Extrair dados de forma robusta
        preco_atual = extrair_valor_escalar(dados['Close'].iloc[-1])
        preco_anterior = extrair_valor_escalar(dados['Close'].iloc[-2]) if len(dados) > 1 else preco_atual
        variacao_mes = ((preco_atual / extrair_valor_escalar(dados['Close'].iloc[-min(30, len(dados)-1)])) - 1) * 100 if len(dados) > 30 else 0
        rsi_atual = extrair_valor_escalar(dados['RSI'].iloc[-1]) if 'RSI' in dados.columns else 50
        volume_atual = extrair_valor_escalar(dados['Volume'].iloc[-1])
        volume_medio = extrair_valor_escalar(dados['Volume'].mean())
        volatilidade = extrair_valor_escalar(dados['Close'].pct_change().std() * 100)
        
        # Análise técnica avançada
        sma_7 = extrair_valor_escalar(dados['SMA_7'].iloc[-1]) if 'SMA_7' in dados.columns else preco_atual
        sma_21 = extrair_valor_escalar(dados['SMA_21'].iloc[-1]) if 'SMA_21' in dados.columns else preco_atual
        macd = extrair_valor_escalar(dados['MACD'].iloc[-1]) if 'MACD' in dados.columns else 0
        bb_width = extrair_valor_escalar(dados['BB_width'].iloc[-1]) if 'BB_width' in dados.columns else 0
        
        prompts = {
            'analise_tecnica': f"""
            Como analista financeiro experiente, analise os indicadores técnicos da ação {ticker}:
            
            📊 DADOS TÉCNICOS:
            • Preço atual: R$ {preco_atual:.2f}
            • RSI (14): {rsi_atual:.2f}
            • Variação mensal: {variacao_mes:.2f}%
            • Volume vs média: {(volume_atual/volume_medio):.2f}x
            • Volatilidade: {volatilidade:.2f}%
            • Médias móveis: SMA7={sma_7:.2f}, SMA21={sma_21:.2f}
            • MACD: {macd:.4f}
            • Largura Bollinger: {bb_width:.2f}
            
            Forneça uma análise técnica CONCISA em 4-5 linhas, identificando:
            - Se está sobrecomprada/sobrevendida (RSI)
            - Tendência das médias móveis
            - Sinais de momentum (MACD)
            - Nível de volatilidade
            
            Use emojis e seja objetivo. Foque nos pontos mais importantes.
            """,

            'tendencia': f"""
            Analise a TENDÊNCIA da ação {ticker} com base nos dados:
            
            📈 ANÁLISE DE TENDÊNCIA:
            • Preço atual: R$ {preco_atual:.2f}
            • SMA 7 dias: R$ {sma_7:.2f}
            • SMA 21 dias: R$ {sma_21:.2f}
            • MACD: {macd:.4f}
            • Posição das médias: {"ALTA" if sma_7 > sma_21 else "BAIXA" if sma_7 < sma_21 else "LATERAL"}
            • Volume relativo: {(volume_atual/volume_medio):.1f}x
            
            Identifique claramente:
            📈 TENDÊNCIA DE ALTA - se indicadores apontam subida
            📉 TENDÊNCIA DE BAIXA - se indicadores apontam queda  
            ➡️ TENDÊNCIA LATERAL - se indefinida
            
            Justifique em 3-4 linhas com base nos indicadores técnicos.
            """,

            'risco_volatilidade': f"""
            Avalie o RISCO e VOLATILIDADE da ação {ticker}:
            
            ⚠️ MÉTRICAS DE RISCO:
            • Volatilidade diária: {volatilidade:.2f}%
            • Accuracy do modelo: {metricas.get('accuracy', 0)*100:.1f}%
            • Volume anômalo: {(volume_atual/volume_medio):.2f}x da média
            • Largura Bollinger: {bb_width:.2f}
            • RSI extremo: {"SIM" if rsi_atual > 70 or rsi_atual < 30 else "NÃO"}
            
            Classifique o risco como:
            🟢 BAIXO RISCO (vol < 2%, RSI 30-70)
            🟡 RISCO MODERADO (vol 2-4%)
            🔴 ALTO RISCO (vol > 4%, RSI extremo)
            
            Explique os principais fatores de risco em 3-4 linhas.
            """,

            'recomendacao_estrategica': f"""
            Com base na análise COMPLETA da ação {ticker}, forneça sugestão estratégica:
            
            💼 CONTEXTO ESTRATÉGICO:
            • Preço: R$ {preco_atual:.2f} (var. mensal: {variacao_mes:+.1f}%)
            • RSI: {rsi_atual:.1f} {"(sobrecomprado)" if rsi_atual > 70 else "(sobrevendido)" if rsi_atual < 30 else "(neutro)"}
            • Tendência: {"ALTA" if sma_7 > sma_21 else "BAIXA"}
            • Volume: {(volume_atual/volume_medio):.1f}x da média
            • Modelo accuracy: {metricas.get('accuracy', 0)*100:.1f}%
            
            Sugira UMA das estratégias:
            💰 ACUMULAR - se sinais técnicos positivos
            ⏸️ AGUARDAR - se sinais indefinidos
            📊 REALIZAR LUCROS - se sobrecomprado
            ⚠️ CAUTELA - se sinais negativos
            
            Justifique a estratégia em 4-5 linhas.
            
            ⚠️ DISCLAIMER: Esta análise é baseada apenas em indicadores técnicos e não constitui recomendação de investimento. Sempre consulte um profissional qualificado.
            """
        }

        # Criar threads para cada insight
        threads = []
        for tipo, prompt in prompts.items():
            t = threading.Thread(target=gerar_insight_async, args=(model, prompt, tipo))
            t.start()
            threads.append(t)
    
    except Exception as e:
        st.error(f"❌ Erro ao preparar insights: {str(e)}")

# === INTERFACE PRINCIPAL ===

st.markdown('<h1 class="dashboard-header">📈 StockAI Predictor v2.0 - Previsão Inteligente</h1>', unsafe_allow_html=True)

# Sidebar melhorada
with st.sidebar:
    st.markdown("### ⚙️ Configurações do Sistema")

    # Seleção de ticker expandida
    tickers_disponiveis = {
        'PETR4.SA': 'Petrobras PN',
        'VALE3.SA': 'Vale ON',
        'ITUB4.SA': 'Itaú Unibanco PN',
        'BBDC4.SA': 'Bradesco PN',
        'ABEV3.SA': 'Ambev ON',
        'WEGE3.SA': 'WEG ON',
        'MGLU3.SA': 'Magazine Luiza ON',
        'RENT3.SA': 'Localiza ON',
        'BPAC11.SA': 'BTG Pactual UNT',
        'PRIO3.SA': 'PetroRio ON',
        'SUZB3.SA': 'Suzano ON',
        'JBSS3.SA': 'JBS ON',
        'B3SA3.SA': 'B3 ON',
        'IRBR3.SA': 'IRB Brasil ON',
        'HAPV3.SA': 'Hapvida ON'
    }

    ticker_selecionado = st.selectbox(
        "🎯 Selecione a Ação",
        options=list(tickers_disponiveis.keys()),
        format_func=lambda x: f"{tickers_disponiveis[x]} ({x})",
        help="Escolha a ação para análise e previsão"
    )

    periodo = st.select_slider(
        "📅 Período de Análise",
        options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
        value='1y',
        help="Período histórico para análise"
    )

    st.markdown("---")

    # Status do modelo melhorado
    st.markdown("### 🤖 Status do Modelo")
    
    model_path = f'models/{ticker_selecionado}_directional_model.keras'
    scaler_path = f'scalers/{ticker_selecionado}_directional_scaler.pkl'
    metrics_path = f'metrics/{ticker_selecionado}_directional_metrics.pkl'
    
    if all(os.path.exists(path) for path in [model_path, scaler_path, metrics_path]):
        try:
            metricas = joblib.load(metrics_path)
            st.markdown(f'<div class="custom-alert alert-success">✅ Modelo Disponível</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                accuracy_val = metricas.get('accuracy', 0) * 100
                precision_val = metricas.get('precision', 0)
                st.markdown(f'''
                <div class="mini-metric-card" style="text-align: center;">
                    <div style="font-size: 0.8rem; color: #888; margin-bottom: 4px;">🎯 Acurácia</div>
                    <div style="font-size: 1.4rem; font-weight: bold; color: #00D4AA;">{accuracy_val:.1f}%</div>
                </div>
                <div class="mini-metric-card" style="text-align: center;">
                    <div style="font-size: 0.8rem; color: #888; margin-bottom: 4px;">📊 F1 Score</div>
                    <div style="font-size: 1.4rem; font-weight: bold; color: #667eea;">{metricas.get('f1_score', 0):.3f}</div>
                </div>
                ''', unsafe_allow_html=True)
            with col2:
                st.markdown(f'''
                <div class="mini-metric-card" style="text-align: center;">
                    <div style="font-size: 0.8rem; color: #888; margin-bottom: 4px;">🔍 Precisão</div>
                    <div style="font-size: 1.4rem; font-weight: bold; color: #f093fb;">{precision_val:.3f}</div>
                </div>
                <div class="mini-metric-card" style="text-align: center;">
                    <div style="font-size: 0.8rem; color: #888; margin-bottom: 4px;">📈 Recall</div>
                    <div style="font-size: 1.4rem; font-weight: bold; color: #FFC107;">{metricas.get('recall', 0):.3f}</div>
                </div>
                ''', unsafe_allow_html=True)
            
            st.caption(f"🕒 Treinado em: {metricas.get('data_treino', 'N/A')}")
            st.caption(f"🧠 Modelo: {metricas.get('modelo_nome', 'N/A')}")
            
        except Exception as e:
            st.markdown(f'<div class="custom-alert alert-error">❌ Erro ao carregar métricas</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="custom-alert alert-warning">⚠️ Modelo não encontrado</div>', unsafe_allow_html=True)
        st.info("Execute o treinamento primeiro!")
    
    st.markdown("---")
    
    # Google API Key para IA
    if GEMINI_AVAILABLE:
        st.markdown("### 🔑 IA Generativa")
        google_api_key = st.text_input(
            "Google API Key",
            type="password",
            help="Insira sua chave da API do Google Gemini"
        )
        if google_api_key and not st.session_state.model_configured:
            model = configurar_gemini(google_api_key)
            if model:
                st.session_state.model_configured = True
                st.session_state.gemini_model = model
                st.success("✅ IA configurada!")
    else:
        st.markdown(f'<div class="custom-alert alert-warning">⚠️ Google Generative AI não instalado</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Configurações de Debug
    st.markdown("### ⚙️ Configurações")
    
    st.session_state.verbose_mode = st.toggle(
        "🔍 Modo Verboso", 
        value=st.session_state.verbose_mode,
        help="Mostra logs detalhados durante o carregamento"
    )
    
    if st.session_state.loading_logs and not st.session_state.verbose_mode:
        with st.expander("📋 Ver Logs de Carregamento", expanded=False):
            for log_type, log_msg in st.session_state.loading_logs[-10:]:  # Últimos 10 logs
                if log_type == "success":
                    st.success(log_msg)
                elif log_type == "warning":
                    st.warning(log_msg)
                elif log_type == "error":
                    st.error(log_msg)
                else:
                    st.info(log_msg)
    
    st.markdown("---")
    
    # Ferramentas de diagnóstico melhoradas
    st.markdown("### 🔧 Ferramentas")
    
    if st.button("🌐 Testar Conectividade", use_container_width=True):
        with st.spinner("Testando..."):
            try:
                test_data = yf.download("AAPL", period="5d", progress=False)
                if test_data is not None and not test_data.empty:
                    st.success("✅ Conectividade OK!")
                else:
                    st.error("❌ Problema de conectividade")
            except Exception as e:
                st.error(f"❌ Erro: {str(e)}")
    
    if st.button("📂 Verificar Estrutura", use_container_width=True):
        st.markdown("#### 📋 Status dos Diretórios:")
        directories = ['models', 'scalers', 'metrics']
        for directory in directories:
            if os.path.exists(directory):
                files = [f for f in os.listdir(directory) if f.endswith(('.keras', '.pkl'))]
                st.success(f"✅ `{directory}/` - {len(files)} arquivos")
            else:
                st.error(f"❌ `{directory}/` não existe")
    
    if st.button("🗑️ Limpar Cache", use_container_width=True):
        st.cache_data.clear()
        st.session_state.model_cache = {}
        st.success("✅ Cache limpo!")
        st.rerun()

# Interface principal
if ticker_selecionado:
    # Verificar se modelo existe
    if not os.path.exists(f'models/{ticker_selecionado}_directional_model.keras'):
        st.markdown(f'''
        <div class="custom-alert alert-error">
            ❌ <strong>Modelo não encontrado para {tickers_disponiveis[ticker_selecionado]}</strong><br>
            Execute o script de treinamento primeiro para criar o modelo.
        </div>
        ''', unsafe_allow_html=True)
        st.stop()
    
    # Carregar dados e modelo
    if st.session_state.verbose_mode:
        with st.spinner('🔄 Carregando dados e modelo...'):
            dados = carregar_dados_ticker(ticker_selecionado, periodo)
            modelo, scaler, metricas, error_msg = carregar_modelo_completo(ticker_selecionado)
    else:
        # Modo silencioso - apenas spinner limpo
        with st.spinner('🔄 Preparando análise...'):
            dados = carregar_dados_ticker(ticker_selecionado, periodo)
            modelo, scaler, metricas, error_msg = carregar_modelo_completo(ticker_selecionado)
        
        # Mostrar apenas resultado final
        if dados is not None and modelo is not None:
            st.success(f"✅ {tickers_disponiveis[ticker_selecionado]} carregado com {len(dados)} registros")
        elif error_msg:
            st.error(f"❌ {error_msg}")
        else:
            st.error("❌ Erro no carregamento dos dados")
    
    if error_msg:
        st.stop()
    
    if dados is None:
        # Mensagem específica para dados insuficientes
        st.markdown(f"""
        <div class="custom-alert alert-warning">
            <h4>⚠️ Dados Insuficientes</h4>
            <p><strong>O período "{periodo}" não forneceu dados suficientes para {tickers_disponiveis[ticker_selecionado]}.</strong></p>
            <p>Possíveis soluções:</p>
            <ul>
                <li>📅 Selecione um período maior (6mo, 1y, 2y, 5y)</li>
                <li>🔄 Verifique se o ticker está correto</li>
                <li>🌐 Verifique sua conexão com a internet</li>
                <li>⏰ Tente novamente em alguns minutos</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Verificar qualidade dos dados
    saude_ok, msg_saude = verificar_saude_dados(dados)
    if not saude_ok:
        st.warning(f"⚠️ Problemas detectados nos dados: {msg_saude}")
        st.stop()

    # Verificação adicional de dados suficientes
    if len(dados) < 10:
        st.error(f"❌ Dados insuficientes para análise: {len(dados)} registros. Tente um período maior.")
        st.stop()
        
    # Verificar se a coluna Close existe e tem dados
    if 'Close' not in dados.columns or dados['Close'].empty:
        st.error("❌ Dados de preço não disponíveis. Tente outro período ou ticker.")
        st.stop()

    # Interface em abas
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard Principal", 
        "🔮 Previsão IA", 
        "🤖 Insights Gemini", 
        "📈 Análise Técnica",
        "🔍 Diagnóstico"
    ])

    with tab1:
        # Métricas principais com cards melhorados
        st.markdown("### 💰 Resumo Financeiro")
        
        # Verificações de segurança antes de calcular métricas
        if len(dados) == 0 or 'Close' not in dados.columns:
            st.error("❌ Dados insuficientes para calcular métricas")
            st.stop()
        
        # Verificar se há dados válidos na coluna Close
        close_data = dados['Close'].dropna()
        if len(close_data) == 0:
            st.error("❌ Não há dados de preço válidos")
            st.stop()
        
        col1, col2, col3, col4, col5 = st.columns(5)

        # Calcular métricas com verificações de segurança
        try:
            preco_atual = extrair_valor_escalar(close_data.iloc[-1])
            preco_anterior = extrair_valor_escalar(close_data.iloc[-2]) if len(close_data) > 1 else preco_atual
            variacao_diaria = ((preco_atual - preco_anterior) / preco_anterior) * 100 if preco_anterior != 0 else 0
        except (IndexError, KeyError):
            st.error("❌ Erro ao calcular variação de preço")
            st.stop()
        
        try:
            volume_atual = extrair_valor_escalar(dados['Volume'].iloc[-1]) if 'Volume' in dados.columns else 0
            volume_medio = extrair_valor_escalar(dados['Volume'].mean()) if 'Volume' in dados.columns else 1
            volume_ratio = volume_atual / volume_medio if volume_medio != 0 else 1
        except:
            volume_atual = 0
            volume_ratio = 1
        
        try:
            rsi_atual = extrair_valor_escalar(dados['RSI'].iloc[-1]) if 'RSI' in dados.columns else 50
        except:
            rsi_atual = 50
        
        try:
            volatilidade = extrair_valor_escalar(close_data.pct_change().std() * 100)
        except:
            volatilidade = 0
        
        # Calcular variação mensal
        try:
            if len(close_data) >= 22:  # ~1 mês de dados
                preco_mes_passado = extrair_valor_escalar(close_data.iloc[-22])
                variacao_mensal = ((preco_atual - preco_mes_passado) / preco_mes_passado) * 100 if preco_mes_passado != 0 else 0
            else:
                variacao_mensal = 0
        except:
            variacao_mensal = 0

        with col1:
            delta_color = "#00D4AA" if variacao_diaria >= 0 else "#FF6B6B"
            delta_symbol = "▲" if variacao_diaria >= 0 else "▼"
            st.markdown(f'''
            <div class="metric-card">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 1.2rem;">💰</span>
                    <span style="margin-left: 8px; font-weight: bold; color: #666;">Preço Atual</span>
                </div>
                <div style="font-size: 2rem; font-weight: bold; color: #2c3e50; margin-bottom: 4px;">
                    R$ {preco_atual:.2f}
                </div>
                <div style="color: {delta_color}; font-weight: bold;">
                    {delta_symbol} {abs(variacao_diaria):.2f}%
                </div>
            </div>
            ''', unsafe_allow_html=True)

        with col2:
            volume_color = "#00D4AA" if volume_ratio >= 1 else "#FFC107"
            st.markdown(f'''
            <div class="metric-card">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 1.2rem;">📊</span>
                    <span style="margin-left: 8px; font-weight: bold; color: #666;">Volume</span>
                </div>
                <div style="font-size: 2rem; font-weight: bold; color: #2c3e50; margin-bottom: 4px;">
                    {volume_atual:,.0f}
                </div>
                <div style="color: {volume_color}; font-weight: bold;">
                    {volume_ratio:.1f}x média
                </div>
            </div>
            ''', unsafe_allow_html=True)

        with col3:
            if rsi_atual > 70:
                rsi_color = "#FF6B6B"
                rsi_status = "⚠️ Sobrecomprado"
            elif rsi_atual < 30:
                rsi_color = "#FF6B6B" 
                rsi_status = "⚠️ Sobrevendido"
            else:
                rsi_color = "#00D4AA"
                rsi_status = "✅ Neutro"
            
            st.markdown(f'''
            <div class="metric-card">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 1.2rem;">📈</span>
                    <span style="margin-left: 8px; font-weight: bold; color: #666;">RSI</span>
                </div>
                <div style="font-size: 2rem; font-weight: bold; color: #2c3e50; margin-bottom: 4px;">
                    {rsi_atual:.1f}
                </div>
                <div style="color: {rsi_color}; font-weight: bold;">
                    {rsi_status}
                </div>
            </div>
            ''', unsafe_allow_html=True)

        with col4:
            if volatilidade > 3:
                vol_color = "#FF6B6B"
                vol_status = "🔴 Alta"
            elif volatilidade > 1.5:
                vol_color = "#FFC107"
                vol_status = "🟡 Média"
            else:
                vol_color = "#00D4AA"
                vol_status = "🟢 Baixa"
            
            st.markdown(f'''
            <div class="metric-card">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 1.2rem;">📉</span>
                    <span style="margin-left: 8px; font-weight: bold; color: #666;">Volatilidade</span>
                </div>
                <div style="font-size: 2rem; font-weight: bold; color: #2c3e50; margin-bottom: 4px;">
                    {volatilidade:.2f}%
                </div>
                <div style="color: {vol_color}; font-weight: bold;">
                    {vol_status}
                </div>
            </div>
            ''', unsafe_allow_html=True)

        with col5:
            if variacao_mensal > 0:
                mensal_color = "#00D4AA"
                mensal_status = "📈 Positiva"
                mensal_symbol = "▲"
            elif variacao_mensal < 0:
                mensal_color = "#FF6B6B"
                mensal_status = "📉 Negativa"
                mensal_symbol = "▼"
            else:
                mensal_color = "#666"
                mensal_status = "➡️ Estável"
                mensal_symbol = "="
            
            st.markdown(f'''
            <div class="metric-card">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 1.2rem;">📅</span>
                    <span style="margin-left: 8px; font-weight: bold; color: #666;">Var. Mensal</span>
                </div>
                <div style="font-size: 2rem; font-weight: bold; color: #2c3e50; margin-bottom: 4px;">
                    {mensal_symbol} {abs(variacao_mensal):.2f}%
                </div>
                <div style="color: {mensal_color}; font-weight: bold;">
                    {mensal_status}
                </div>
            </div>
            ''', unsafe_allow_html=True)

        # Gráfico principal melhorado
        st.markdown("### 📈 Análise Gráfica Avançada")

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(
                f"Preços e Indicadores - {tickers_disponiveis[ticker_selecionado]}", 
                "Volume de Negociação", 
                "RSI (14 períodos)"
            )
        )

        # Candlestick melhorado
        fig.add_trace(
            go.Candlestick(
                x=dados.index,
                open=dados['Open'],
                high=dados['High'],
                low=dados['Low'],
                close=dados['Close'],
                name="Preço",
                increasing_line_color='#00D4AA',
                decreasing_line_color='#FF6B6B',
                increasing_fillcolor='#00D4AA',
                decreasing_fillcolor='#FF6B6B'
            ),
            row=1, col=1
        )

        # Médias móveis com cores modernas
        if 'SMA_7' in dados.columns:
            fig.add_trace(
                go.Scatter(
                    x=dados.index,
                    y=dados['SMA_7'],
                    name="SMA 7",
                    line=dict(color='#667eea', width=2)
                ),
                row=1, col=1
            )

        if 'SMA_21' in dados.columns:
            fig.add_trace(
                go.Scatter(
                    x=dados.index,
                    y=dados['SMA_21'],
                    name="SMA 21",
                    line=dict(color='#f093fb', width=2)
                ),
                row=1, col=1
            )

        # Bollinger Bands
        if all(col in dados.columns for col in ['BB_upper', 'BB_lower']):
            fig.add_trace(
                go.Scatter(
                    x=dados.index,
                    y=dados['BB_upper'],
                    name="BB Superior",
                    line=dict(color='rgba(100, 100, 100, 0.3)', width=1),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=dados.index,
                    y=dados['BB_lower'],
                    name="BB Inferior",
                    line=dict(color='rgba(100, 100, 100, 0.3)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(100, 100, 100, 0.1)',
                    showlegend=False
                ),
                row=1, col=1
            )

        # Volume com cores baseadas no preço
        colors = []
        for i in range(len(dados)):
            close_val = extrair_valor_escalar(dados['Close'].iloc[i])
            open_val = extrair_valor_escalar(dados['Open'].iloc[i])
            colors.append('#00D4AA' if close_val >= open_val else '#FF6B6B')

        fig.add_trace(
            go.Bar(
                x=dados.index,
                y=dados['Volume'],
                name="Volume",
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )

        # RSI
        if 'RSI' in dados.columns:
            fig.add_trace(
                go.Scatter(
                    x=dados.index,
                    y=dados['RSI'],
                    name='RSI',
                    line=dict(color='#667eea', width=2)
                ),
                row=3, col=1
            )

            # Linhas de referência RSI
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)

        fig.update_layout(
            template="plotly_white",
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=50, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Resumo do dia
        st.markdown("### 📋 Resumo do Dia")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **📊 Dados de Preço:**
            - **Abertura:** R$ {extrair_valor_escalar(dados['Open'].iloc[-1]):.2f}
            - **Máxima:** R$ {extrair_valor_escalar(dados['High'].iloc[-1]):.2f}
            - **Mínima:** R$ {extrair_valor_escalar(dados['Low'].iloc[-1]):.2f}
            - **Fechamento:** R$ {preco_atual:.2f}
            - **Variação:** {variacao_diaria:+.2f}%
            """)
        
        with col2:
            amplitude = extrair_valor_escalar(dados['High'].iloc[-1]) - extrair_valor_escalar(dados['Low'].iloc[-1])
            st.markdown(f"""
            **📈 Estatísticas:**
            - **Volume:** {volume_atual:,.0f}
            - **Amplitude:** R$ {amplitude:.2f}
            - **RSI:** {rsi_atual:.1f}
            - **Volatilidade:** {volatilidade:.2f}%
            """)

    with tab2:
        st.markdown("### 🔮 Previsão Direcional com IA")

        if modelo is not None and scaler is not None:
            try:
                # Preparar dados para previsão
                features = metricas.get('feature_names', [])
                janela = metricas.get('janela', 30)

                if len(dados) >= janela:
                    # Preparar features
                    dados_prep = preparar_dados_previsao(dados, metricas)
                    
                    if dados_prep is not None:
                        ultimos_dados = dados_prep[-janela:]
                        
                        # Escalar dados
                        ultimos_dados_norm = scaler.transform(ultimos_dados)
                        
                        # Fazer previsão
                        X_pred = ultimos_dados_norm.reshape(1, janela, ultimos_dados_norm.shape[1])
                        
                        with st.spinner("🤖 Processando previsão..."):
                            previsao_proba = modelo.predict(X_pred, verbose=0)[0, 0]
                            previsao_classe = 1 if previsao_proba > 0.5 else 0
                            confianca = abs(previsao_proba - 0.5) * 2

                        # Display da previsão com design moderno
                        col1, col2, col3 = st.columns([1, 2, 1])

                        with col2:
                            if previsao_classe == 1:
                                st.markdown(f'''
                                <div class="metric-card" style="text-align: center; background: linear-gradient(135deg, #00D4AA, #00B894);">
                                    <h1 style="color: white; margin: 0;">📈 TENDÊNCIA DE ALTA</h1>
                                    <h2 style="color: white; margin: 10px 0;">Probabilidade: {previsao_proba:.1%}</h2>
                                    <!--<h3 style="color: white; margin: 0;">Confiança: {confianca:.1%}</h3>-->
                                </div>
                                ''', unsafe_allow_html=True)
                            else:
                                st.markdown(f'''
                                <div class="metric-card" style="text-align: center; background: linear-gradient(135deg, #FF6B6B, #E55656);">
                                    <h1 style="color: white; margin: 0;">📉 TENDÊNCIA DE BAIXA</h1>
                                    <h2 style="color: white; margin: 10px 0;">Probabilidade: {1-previsao_proba:.1%}</h2>
                                    <!--<h3 style="color: white; margin: 0;">Confiança: {confianca:.1%}</h3>-->
                                </div>
                                ''', unsafe_allow_html=True)

                        # Métricas do modelo
                        st.markdown("### 📊 Performance do Modelo")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            accuracy = metricas.get('accuracy', 0)
                            color = "#00D4AA" if accuracy > 0.7 else "#FFC107" if accuracy > 0.6 else "#FF6B6B"
                            emoji = "🟢" if accuracy > 0.7 else "🟡" if accuracy > 0.6 else "🔴"
                            st.markdown(f'''
                            <div class="metric-card">
                                <div style="text-align: center;">
                                    <div style="font-size: 1.5rem; margin-bottom: 8px;">{emoji}</div>
                                    <div style="font-size: 1.8rem; font-weight: bold; color: {color}; margin-bottom: 4px;">
                                        {accuracy:.1%}
                                    </div>
                                    <div style="color: #666; font-weight: bold;">Acurácia</div>
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        with col2:
                            precision = metricas.get('precision', 0)
                            st.markdown(f'''
                            <div class="metric-card">
                                <div style="text-align: center;">
                                    <div style="font-size: 1.5rem; margin-bottom: 8px;">🎯</div>
                                    <div style="font-size: 1.8rem; font-weight: bold; color: #667eea; margin-bottom: 4px;">
                                        {precision:.3f}
                                    </div>
                                    <div style="color: #666; font-weight: bold;">Precisão</div>
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        with col3:
                            recall = metricas.get('recall', 0)
                            st.markdown(f'''
                            <div class="metric-card">
                                <div style="text-align: center;">
                                    <div style="font-size: 1.5rem; margin-bottom: 8px;">📈</div>
                                    <div style="font-size: 1.8rem; font-weight: bold; color: #00D4AA; margin-bottom: 4px;">
                                        {recall:.3f}
                                    </div>
                                    <div style="color: #666; font-weight: bold;">Recall</div>
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        with col4:
                            f1_score = metricas.get('f1_score', 0)
                            st.markdown(f'''
                            <div class="metric-card">
                                <div style="text-align: center;">
                                    <div style="font-size: 1.5rem; margin-bottom: 8px;">⚖️</div>
                                    <div style="font-size: 1.8rem; font-weight: bold; color: #f093fb; margin-bottom: 4px;">
                                        {f1_score:.3f}
                                    </div>
                                    <div style="color: #666; font-weight: bold;">F1 Score</div>
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)

                        # Gráfico de previsão
                        st.markdown("### 📈 Visualização da Previsão")

                        # Últimos 30 dias
                        periodo_viz = min(30, len(dados))
                        dados_viz = dados['Close'].iloc[-periodo_viz:].copy()
                        
                        fig_prev = go.Figure()

                        # Dados históricos
                        fig_prev.add_trace(go.Scatter(
                            x=dados_viz.index,
                            y=dados_viz.values,
                            mode='lines+markers',
                            name='Histórico',
                            line=dict(color='#667eea', width=3),
                            marker=dict(size=4)
                        ))

                        # Ponto atual
                        ultimo_preco = extrair_valor_escalar(dados_viz.iloc[-1])
                        fig_prev.add_trace(go.Scatter(
                            x=[dados_viz.index[-1]],
                            y=[ultimo_preco],
                            mode='markers',
                            name='Preço Atual',
                            marker=dict(
                                size=15,
                                color='gold',
                                symbol='star',
                                line=dict(width=2, color='black')
                            )
                        ))

                        # Sinal de previsão
                        cor_sinal = '#00D4AA' if previsao_classe == 1 else '#FF6B6B'
                        simbolo_sinal = 'triangle-up' if previsao_classe == 1 else 'triangle-down'
                        texto_sinal = f"📈 {previsao_proba:.1%}" if previsao_classe == 1 else f"📉 {1-previsao_proba:.1%}"
                        
                        fig_prev.add_annotation(
                            x=dados_viz.index[-1],
                            y=ultimo_preco,
                            text=texto_sinal,
                            showarrow=True,
                            arrowhead=2,
                            arrowcolor=cor_sinal,
                            arrowwidth=3,
                            arrowsize=2,
                            ax=50 if previsao_classe == 1 else -50,
                            ay=-50,
                            font=dict(size=14, color=cor_sinal, family="Arial Black"),
                            bgcolor="white",
                            bordercolor=cor_sinal,
                            borderwidth=2
                        )

                        fig_prev.update_layout(
                            template="plotly_white",
                            height=500,
                            title=f"Previsão para {tickers_disponiveis[ticker_selecionado]} - Próximo Pregão",
                            xaxis_title="Data",
                            yaxis_title="Preço (R$)",
                            showlegend=True,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )

                        st.plotly_chart(fig_prev, use_container_width=True)

                        # Interpretação da previsão
                        st.markdown("### 🧠 Interpretação da IA")
                        
                        if confianca > 0.7:
                            conf_emoji = "🟢"
                            conf_text = "ALTA CONFIANÇA"
                        elif confianca > 0.4:
                            conf_emoji = "🟡"
                            conf_text = "CONFIANÇA MODERADA"
                        else:
                            conf_emoji = "🔴"
                            conf_text = "BAIXA CONFIANÇA"

                        st.markdown(f"""
                        **{conf_emoji} Nível de Confiança: {conf_text}**
                        
                        **Análise:** O modelo prevê uma **{'ALTA' if previsao_classe == 1 else 'BAIXA'}** 
                        com probabilidade de **{max(previsao_proba, 1-previsao_proba):.1%}**.
                        
                        **Fatores considerados:**
                        - 📊 {len(metricas.get('feature_names', []))} indicadores técnicos
                        - 📈 Dados de {janela} períodos anteriores
                        - 🌍 Correlações com índices de mercado
                        - 🤖 Modelo treinado: {metricas.get('modelo_nome', 'N/A')}
                        
                        **⚠️ Importante:** Esta previsão é baseada em padrões históricos e não garante resultados futuros. 
                        Use sempre em conjunto com outras análises e consulte profissionais qualificados.
                        """)
                        
                        # Card adicional com design
                        st.markdown("""
                        <div class="insight-card">
                        <h4>💡 Como interpretar esta previsão:</h4>
                        <ul>
                        <li><strong>Alta Confiança (>70%):</strong> Modelo muito seguro da previsão</li>
                        <li><strong>Confiança Moderada (40-70%):</strong> Previsão com incerteza</li>
                        <li><strong>Baixa Confiança (<40%):</strong> Cenário indefinido, aguarde mais dados</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)

                    else:
                        st.error("❌ Erro ao preparar dados para previsão")
                else:
                    st.error(f"❌ Dados insuficientes. Necessário: {janela} períodos, disponível: {len(dados)}")

            except Exception as e:
                st.error(f"❌ Erro na previsão: {str(e)}")
                st.code(f"Detalhes do erro: {type(e).__name__}: {str(e)}")

        else:
            st.error("❌ Modelo não carregado corretamente")

    with tab3:
        st.markdown("### 🤖 Insights Gerados por IA")

        if GEMINI_AVAILABLE and st.session_state.model_configured:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Gerar Análise Completa com IA", use_container_width=True):
                    st.session_state.insights_data = {}
                    model = st.session_state.gemini_model
                    
                    with st.spinner("🧠 IA analisando dados..."):
                        iniciar_geracao_insights(model, ticker_selecionado, dados, metricas)
                        
                        # Aguardar insights
                        max_wait = 30  # 30 segundos
                        start_time = time.time()
                        
                        while len(st.session_state.insights_data) < 4 and (time.time() - start_time) < max_wait:
                            while not st.session_state.insights_queue.empty():
                                tipo, conteudo = st.session_state.insights_queue.get()
                                st.session_state.insights_data[tipo] = conteudo
                            time.sleep(0.5)

            # Display dos insights em layout melhorado
            if st.session_state.insights_data:
                col1, col2 = st.columns(2)

                with col1:
                    if 'analise_tecnica' in st.session_state.insights_data:
                        st.markdown("#### 📊 Análise Técnica")
                        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                        st.markdown(st.session_state.insights_data['analise_tecnica'])
                        st.markdown('</div>', unsafe_allow_html=True)

                    if 'risco_volatilidade' in st.session_state.insights_data:
                        st.markdown("#### ⚠️ Risco e Volatilidade")
                        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                        st.markdown(st.session_state.insights_data['risco_volatilidade'])
                        st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    if 'tendencia' in st.session_state.insights_data:
                        st.markdown("#### 📈 Análise de Tendência")
                        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                        st.markdown(st.session_state.insights_data['tendencia'])
                        st.markdown('</div>', unsafe_allow_html=True)

                    if 'recomendacao_estrategica' in st.session_state.insights_data:
                        st.markdown("#### 💡 Sugestão Estratégica")
                        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                        st.markdown(st.session_state.insights_data['recomendacao_estrategica'])
                        st.markdown('</div>', unsafe_allow_html=True)

                # Aguardar mais insights se necessário
                if len(st.session_state.insights_data) < 4:
                    with st.spinner("🔄 Finalizando análise..."):
                        time.sleep(2)
                        st.rerun()
            
            else:
                st.info("👆 Clique no botão acima para gerar insights personalizados com IA")
                
                # Exemplo de insights
                st.markdown("""
                ### 🌟 Exemplo de Insights que você receberá:
                
                - 📊 **Análise Técnica:** Interpretação de RSI, MACD, Médias Móveis
                - 📈 **Tendência:** Direção provável baseada em indicadores  
                - ⚠️ **Risco:** Avaliação de volatilidade e confiança do modelo
                - 💡 **Estratégia:** Sugestões de ação (acumular, aguardar, etc.)
                
                *Powered by Google Gemini AI*
                """)

        else:
            st.warning("🔑 Configure sua Google API Key na barra lateral para ativar insights de IA")
            st.info("Obtenha uma chave gratuita em: https://makersuite.google.com/app/apikey")

    with tab4:
        st.markdown("### 📈 Análise Técnica Detalhada")

        # Indicadores em layout organizado
        st.markdown("#### 📊 Indicadores Principais")
        
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("**🏷️ Médias Móveis**")
            if 'SMA_7' in dados.columns:
                sma7_val = extrair_valor_escalar(dados['SMA_7'].iloc[-1])
                st.markdown(f'''
                <div class="mini-metric-card">
                    <div style="font-size: 0.9rem; color: #666; margin-bottom: 4px;">SMA 7</div>
                    <div style="font-size: 1.3rem; font-weight: bold; color: #2c3e50;">R$ {sma7_val:.2f}</div>
                </div>
                ''', unsafe_allow_html=True)
            if 'SMA_21' in dados.columns:
                sma21_val = extrair_valor_escalar(dados['SMA_21'].iloc[-1])
                st.markdown(f'''
                <div class="mini-metric-card">
                    <div style="font-size: 0.9rem; color: #666; margin-bottom: 4px;">SMA 21</div>
                    <div style="font-size: 1.3rem; font-weight: bold; color: #2c3e50;">R$ {sma21_val:.2f}</div>
                </div>
                ''', unsafe_allow_html=True)
            if 'EMA_9' in dados.columns:
                ema9_val = extrair_valor_escalar(dados['EMA_9'].iloc[-1])
                st.markdown(f'''
                <div class="mini-metric-card">
                    <div style="font-size: 0.9rem; color: #666; margin-bottom: 4px;">EMA 9</div>
                    <div style="font-size: 1.3rem; font-weight: bold; color: #2c3e50;">R$ {ema9_val:.2f}</div>
                </div>
                ''', unsafe_allow_html=True)

        with col2:
            st.markdown("**📈 Momentum**")
            if 'RSI' in dados.columns:
                rsi_val = extrair_valor_escalar(dados['RSI'].iloc[-1])
                st.markdown(f'''
                <div class="mini-metric-card">
                    <div style="font-size: 0.9rem; color: #666; margin-bottom: 4px;">RSI (14)</div>
                    <div style="font-size: 1.3rem; font-weight: bold; color: #2c3e50;">{rsi_val:.2f}</div>
                </div>
                ''', unsafe_allow_html=True)
            if 'MACD' in dados.columns:
                macd_val = extrair_valor_escalar(dados['MACD'].iloc[-1])
                st.markdown(f'''
                <div class="mini-metric-card">
                    <div style="font-size: 0.9rem; color: #666; margin-bottom: 4px;">MACD</div>
                    <div style="font-size: 1.3rem; font-weight: bold; color: #2c3e50;">{macd_val:.4f}</div>
                </div>
                ''', unsafe_allow_html=True)
            if 'MACD_signal' in dados.columns:
                macd_signal_val = extrair_valor_escalar(dados['MACD_signal'].iloc[-1])
                st.markdown(f'''
                <div class="mini-metric-card">
                    <div style="font-size: 0.9rem; color: #666; margin-bottom: 4px;">MACD Signal</div>
                    <div style="font-size: 1.3rem; font-weight: bold; color: #2c3e50;">{macd_signal_val:.4f}</div>
                </div>
                ''', unsafe_allow_html=True)

        with col3:
            st.markdown("**📉 Bollinger Bands**")
            if 'BB_upper' in dados.columns:
                bb_upper_val = extrair_valor_escalar(dados['BB_upper'].iloc[-1])
                st.markdown(f'''
                <div class="mini-metric-card">
                    <div style="font-size: 0.9rem; color: #666; margin-bottom: 4px;">BB Superior</div>
                    <div style="font-size: 1.3rem; font-weight: bold; color: #2c3e50;">R$ {bb_upper_val:.2f}</div>
                </div>
                ''', unsafe_allow_html=True)
            if 'BB_lower' in dados.columns:
                bb_lower_val = extrair_valor_escalar(dados['BB_lower'].iloc[-1])
                st.markdown(f'''
                <div class="mini-metric-card">
                    <div style="font-size: 0.9rem; color: #666; margin-bottom: 4px;">BB Inferior</div>
                    <div style="font-size: 1.3rem; font-weight: bold; color: #2c3e50;">R$ {bb_lower_val:.2f}</div>
                </div>
                ''', unsafe_allow_html=True)
            if 'BB_width' in dados.columns:
                bb_width_val = extrair_valor_escalar(dados['BB_width'].iloc[-1])
                st.markdown(f'''
                <div class="mini-metric-card">
                    <div style="font-size: 0.9rem; color: #666; margin-bottom: 4px;">Largura BB</div>
                    <div style="font-size: 1.3rem; font-weight: bold; color: #2c3e50;">{bb_width_val:.2f}</div>
                </div>
                ''', unsafe_allow_html=True)

        with col4:
            st.markdown("**📊 Volume & Outros**")
            if 'Volume_ratio' in dados.columns:
                vol_ratio_val = extrair_valor_escalar(dados['Volume_ratio'].iloc[-1])
                st.markdown(f'''
                <div class="mini-metric-card">
                    <div style="font-size: 0.9rem; color: #666; margin-bottom: 4px;">Volume Ratio</div>
                    <div style="font-size: 1.3rem; font-weight: bold; color: #2c3e50;">{vol_ratio_val:.2f}x</div>
                </div>
                ''', unsafe_allow_html=True)
            volatilidade_val = extrair_valor_escalar(dados['Close'].pct_change().std() * 100)
            st.markdown(f'''
            <div class="mini-metric-card">
                <div style="font-size: 0.9rem; color: #666; margin-bottom: 4px;">Volatilidade</div>
                <div style="font-size: 1.3rem; font-weight: bold; color: #2c3e50;">{volatilidade_val:.2f}%</div>
            </div>
            ''', unsafe_allow_html=True)
            amplitude_val = (extrair_valor_escalar(dados['High'].iloc[-1]) - extrair_valor_escalar(dados['Low'].iloc[-1]))
            st.markdown(f'''
            <div class="mini-metric-card">
                <div style="font-size: 0.9rem; color: #666; margin-bottom: 4px;">Amplitude H-L</div>
                <div style="font-size: 1.3rem; font-weight: bold; color: #2c3e50;">{amplitude_val:.2f}</div>
            </div>
            ''', unsafe_allow_html=True)

        # Gráficos de indicadores
        st.markdown("#### 📊 Visualização de Indicadores")

        # RSI com zonas
        col1, col2 = st.columns(2)
        
        with col1:
            if 'RSI' in dados.columns:
                fig_rsi = go.Figure()
                
                fig_rsi.add_trace(go.Scatter(
                    x=dados.index,
                    y=dados['RSI'],
                    name='RSI',
                    line=dict(color='#667eea', width=3)
                ))

                # Zonas de RSI
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Sobrecomprado (70)")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Sobrevendido (30)")
                fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutro (50)")

                # Área de sobrecompra/sobrevenda
                fig_rsi.add_shape(type="rect", x0=dados.index[0], x1=dados.index[-1], y0=70, y1=100, 
                                fillcolor="red", opacity=0.1, line_width=0)
                fig_rsi.add_shape(type="rect", x0=dados.index[0], x1=dados.index[-1], y0=0, y1=30, 
                                fillcolor="green", opacity=0.1, line_width=0)

                fig_rsi.update_layout(
                    template="plotly_white",
                    height=300,
                    title="RSI (14) - Índice de Força Relativa",
                    yaxis_title="RSI",
                    yaxis=dict(range=[0, 100])
                )

                st.plotly_chart(fig_rsi, use_container_width=True)

        with col2:
            # MACD
            if all(col in dados.columns for col in ['MACD', 'MACD_signal']):
                fig_macd = go.Figure()
                
                fig_macd.add_trace(go.Scatter(
                    x=dados.index,
                    y=dados['MACD'],
                    name='MACD',
                    line=dict(color='#00d4aa', width=2)
                ))

                fig_macd.add_trace(go.Scatter(
                    x=dados.index,
                    y=dados['MACD_signal'],
                    name='Signal',
                    line=dict(color='#f093fb', width=2)
                ))

                # Histograma MACD
                macd_hist = dados['MACD'] - dados['MACD_signal']
                colors_macd = ['#00d4aa' if val >= 0 else '#ff6b6b' for val in macd_hist]

                fig_macd.add_trace(go.Bar(
                    x=dados.index,
                    y=macd_hist,
                    name='Histograma',
                    marker_color=colors_macd,
                    opacity=0.7
                ))

                fig_macd.update_layout(
                    template="plotly_white",
                    height=300,
                    title="MACD - Convergência e Divergência de Médias Móveis",
                    yaxis_title="Valor"
                )

                st.plotly_chart(fig_macd, use_container_width=True)

        # Análise de suporte e resistência
        st.markdown("#### 🎯 Suporte e Resistência")
        
        # Calcular níveis de suporte e resistência
        periodo_sr = min(50, len(dados))
        dados_sr = dados.iloc[-periodo_sr:]
        
        resistencia = extrair_valor_escalar(dados_sr['High'].max())
        suporte = extrair_valor_escalar(dados_sr['Low'].min())
        preco_atual = extrair_valor_escalar(dados['Close'].iloc[-1])
        
        # Posição do preço
        if suporte != resistencia:
            posicao_pct = ((preco_atual - suporte) / (resistencia - suporte)) * 100
        else:
            posicao_pct = 50
        
        # Calcular amplitude
        amplitude_sr = resistencia - suporte
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'''
            <div style="background: rgba(255, 100, 100, 0.1); padding: 1rem; border-radius: 10px; text-align: center;">
                <div style="font-size: 1.2rem; margin-bottom: 8px;">🔴</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: #e74c3c; margin-bottom: 4px;">
                    R$ {resistencia:.2f}
                </div>
                <div style="color: #666; font-weight: bold;">Resistência</div>
            </div>
            ''', unsafe_allow_html=True)
        with col2:
            st.markdown(f'''
            <div style="background: rgba(100, 255, 100, 0.1); padding: 1rem; border-radius: 10px; text-align: center;">
                <div style="font-size: 1.2rem; margin-bottom: 8px;">🟢</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: #27ae60; margin-bottom: 4px;">
                    R$ {suporte:.2f}
                </div>
                <div style="color: #666; font-weight: bold;">Suporte</div>
            </div>
            ''', unsafe_allow_html=True)
        with col3:
            pos_color = "#e74c3c" if posicao_pct > 80 else "#27ae60" if posicao_pct < 20 else "#f39c12"
            st.markdown(f'''
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; text-align: center;">
                <div style="font-size: 1.2rem; margin-bottom: 8px;">📊</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: {pos_color}; margin-bottom: 4px;">
                    {posicao_pct:.1f}%
                </div>
                <div style="color: #666; font-weight: bold;">Posição</div>
            </div>
            ''', unsafe_allow_html=True)
        with col4:
            st.markdown(f'''
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; text-align: center;">
                <div style="font-size: 1.2rem; margin-bottom: 8px;">📏</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: #3498db; margin-bottom: 4px;">
                    R$ {amplitude_sr:.2f}
                </div>
                <div style="color: #666; font-weight: bold;">Amplitude S/R</div>
            </div>
            ''', unsafe_allow_html=True)

        # Interpretação dos indicadores
        st.markdown("#### 🧠 Interpretação Técnica")
        
        interpretacoes = []
        
        # RSI
        if 'RSI' in dados.columns:
            rsi_atual = extrair_valor_escalar(dados['RSI'].iloc[-1])
            if rsi_atual > 70:
                interpretacoes.append("🔴 **RSI:** Ativo em zona de sobrecompra, possível correção")
            elif rsi_atual < 30:
                interpretacoes.append("🟢 **RSI:** Ativo em zona de sobrevenda, possível recuperação")
            else:
                interpretacoes.append("🟡 **RSI:** Ativo em zona neutra, sem sinais extremos")
        
        # Médias móveis
        if 'SMA_7' in dados.columns and 'SMA_21' in dados.columns:
            sma7 = extrair_valor_escalar(dados['SMA_7'].iloc[-1])
            sma21 = extrair_valor_escalar(dados['SMA_21'].iloc[-1])
            if sma7 > sma21:
                interpretacoes.append("📈 **Médias Móveis:** Tendência de alta (SMA7 > SMA21)")
            elif sma7 < sma21:
                interpretacoes.append("📉 **Médias Móveis:** Tendência de baixa (SMA7 < SMA21)")
            else:
                interpretacoes.append("➡️ **Médias Móveis:** Movimento lateral")
        
        # Volume
        volume_atual = extrair_valor_escalar(dados['Volume'].iloc[-1])
        volume_medio = extrair_valor_escalar(dados['Volume'].mean())
        if volume_atual > volume_medio * 1.5:
            interpretacoes.append("📊 **Volume:** Alto volume confirma o movimento")
        elif volume_atual < volume_medio * 0.5:
            interpretacoes.append("📊 **Volume:** Baixo volume, movimento sem convicção")
        else:
            interpretacoes.append("📊 **Volume:** Volume normal")
        
        # Posição S/R
        if posicao_pct > 80:
            interpretacoes.append("🎯 **S/R:** Próximo da resistência, cuidado com reversão")
        elif posicao_pct < 20:
            interpretacoes.append("🎯 **S/R:** Próximo do suporte, possível sustentação")
        else:
            interpretacoes.append("🎯 **S/R:** No meio do canal, sem pressão de S/R")
        
        for interpretacao in interpretacoes:
            st.markdown(f"- {interpretacao}")

    with tab5:
        st.markdown("### 🔍 Diagnóstico Completo do Sistema")
        
        # Status geral
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Status dos Componentes")
            
            # Verificar componentes
            components_status = []
            
            # Dados
            if dados is not None and not dados.empty:
                components_status.append(("✅", "Dados", f"{len(dados)} registros carregados"))
            else:
                components_status.append(("❌", "Dados", "Falha ao carregar"))
            
            # Modelo
            if modelo is not None:
                components_status.append(("✅", "Modelo", "Carregado com sucesso"))
            else:
                components_status.append(("❌", "Modelo", "Falha ao carregar"))
            
            # Scaler
            if scaler is not None:
                components_status.append(("✅", "Scaler", "Carregado com sucesso"))
            else:
                components_status.append(("❌", "Scaler", "Falha ao carregar"))
            
            # Métricas
            if metricas is not None:
                components_status.append(("✅", "Métricas", f"Modelo: {metricas.get('modelo_nome', 'N/A')}"))
            else:
                components_status.append(("❌", "Métricas", "Falha ao carregar"))
            
            # IA
            if st.session_state.model_configured:
                components_status.append(("✅", "IA Gemini", "Configurada e pronta"))
            else:
                components_status.append(("⚠️", "IA Gemini", "Não configurada"))
            
            for status, component, description in components_status:
                st.markdown(f"{status} **{component}:** {description}")
        
        with col2:
            st.markdown("#### 📊 Informações do Modelo")
            
            if metricas:
                st.markdown(f"""
                **📈 Performance:**
                - Acurácia: {metricas.get('accuracy', 0)*100:.2f}%
                - Precisão: {metricas.get('precision', 0):.3f}
                - Recall: {metricas.get('recall', 0):.3f}
                - F1 Score: {metricas.get('f1_score', 0):.3f}
                
                **🔧 Configurações:**
                - Modelo: {metricas.get('modelo_nome', 'N/A')}
                - Janela: {metricas.get('janela', 'N/A')} períodos
                - Features: {metricas.get('num_features', 'N/A')}
                
                **📅 Informações:**
                - Treinado em: {metricas.get('data_treino', 'N/A')}
                - Dados treino: {metricas.get('dados_treino', 'N/A'):,}
                - Dados teste: {metricas.get('dados_teste', 'N/A'):,}
                """)
        
        # Informações dos dados
        st.markdown("#### 📊 Análise dos Dados Carregados")
        
        if dados is not None:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📈 Estatísticas Básicas**")
                preco_min = extrair_valor_escalar(dados['Close'].min())
                preco_max = extrair_valor_escalar(dados['Close'].max())
                preco_medio = extrair_valor_escalar(dados['Close'].mean())
                
                st.markdown(f"""
                - **Período:** {dados.index[0].strftime('%Y-%m-%d')} a {dados.index[-1].strftime('%Y-%m-%d')}
                - **Registros:** {len(dados):,}
                - **Preço mín:** R$ {preco_min:.2f}
                - **Preço máx:** R$ {preco_max:.2f}
                - **Preço médio:** R$ {preco_medio:.2f}
                """)
            
            with col2:
                st.markdown("**📊 Qualidade dos Dados**")
                
                # Calcular estatísticas de qualidade
                total_cells = len(dados) * len(dados.columns)
                nan_count = dados.isnull().sum().sum()
                nan_pct = (nan_count / total_cells) * 100
                
                zero_prices = (dados['Close'] == 0).sum()
                zero_pct = (zero_prices / len(dados)) * 100
                
                st.markdown(f"""
                - **Colunas:** {len(dados.columns)}
                - **NaN total:** {nan_count:,} ({nan_pct:.2f}%)
                - **Preços zero:** {zero_prices} ({zero_pct:.2f}%)
                - **Volatilidade:** {extrair_valor_escalar(dados['Close'].pct_change().std() * 100):.2f}%
                """)
            
            with col3:
                st.markdown("**🔧 Indicadores Calculados**")
                
                # Contar indicadores por categoria
                colunas = dados.columns.tolist()
                
                ma_count = len([col for col in colunas if 'SMA_' in col or 'EMA_' in col])
                momentum_count = len([col for col in colunas if any(x in col for x in ['RSI', 'MACD', 'Williams', 'ROC'])])
                volume_count = len([col for col in colunas if any(x in col for x in ['Volume', 'OBV', 'MFI', 'CMF'])])
                volatility_count = len([col for col in colunas if any(x in col for x in ['BB_', 'ATR', 'KC_'])])
                
                st.markdown(f"""
                - **Médias móveis:** {ma_count}
                - **Momentum:** {momentum_count}
                - **Volume:** {volume_count}
                - **Volatilidade:** {volatility_count}
                - **Total features:** {len(colunas)}
                """)
        
        # Log de features importantes
        if metricas and 'feature_names' in metricas:
            st.markdown("#### 🎯 Features Selecionadas pelo Modelo")
            
            features = metricas['feature_names']
            st.markdown(f"**Total de features selecionadas:** {len(features)}")
            
            # Agrupar features por categoria
            categories = {
                '📈 Médias Móveis': [f for f in features if any(x in f for x in ['SMA_', 'EMA_', 'WMA_', 'KAMA_'])],
                '⚡ Momentum': [f for f in features if any(x in f for x in ['RSI', 'MACD', 'Williams', 'ROC', 'Stoch', 'CCI'])],
                '📊 Volume': [f for f in features if any(x in f for x in ['Volume', 'OBV', 'MFI', 'CMF', 'FI', 'VPT'])],
                '📉 Volatilidade': [f for f in features if any(x in f for x in ['BB_', 'ATR', 'KC_', 'DC_', 'UI'])],
                '📈 Trend': [f for f in features if any(x in f for x in ['ADX', 'Aroon', 'PSAR', 'Ichimoku', 'STC', 'Trix'])],
                '🕒 Temporal': [f for f in features if any(x in f for x in ['Day_', 'Week_', 'Month', 'Quarter', 'Is_'])],
                '📊 Estatísticas': [f for f in features if any(x in f for x in ['Rolling_', 'Log_Return', 'Cum_'])],
                '🎯 Outros': []
            }
            
            # Classificar features não categorizadas
            categorized = set()
            for cat_features in categories.values():
                categorized.update(cat_features)
            
            categories['🎯 Outros'] = [f for f in features if f not in categorized]
            
            # Mostrar em colunas
            col1, col2 = st.columns(2)
            
            cats_items = list(categories.items())
            mid_point = len(cats_items) // 2
            
            with col1:
                for cat, feats in cats_items[:mid_point]:
                    if feats:
                        with st.expander(f"{cat} ({len(feats)})"):
                            for feat in feats[:10]:  # Mostrar apenas os primeiros 10
                                st.markdown(f"• {feat}")
                            if len(feats) > 10:
                                st.markdown(f"... e mais {len(feats) - 10}")
            
            with col2:
                for cat, feats in cats_items[mid_point:]:
                    if feats:
                        with st.expander(f"{cat} ({len(feats)})"):
                            for feat in feats[:10]:
                                st.markdown(f"• {feat}")
                            if len(feats) > 10:
                                st.markdown(f"... e mais {len(feats) - 10}")
        
        # Teste de conectividade avançado
        st.markdown("#### 🌐 Teste de Conectividade Avançado")
        
        if st.button("🔍 Executar Teste Completo", use_container_width=True):
            test_results = []
            
            with st.spinner("Executando testes de conectividade..."):
                # Teste 1: Yahoo Finance
                try:
                    test_data = yf.download("AAPL", period="1d", progress=False)
                    if not test_data.empty:
                        test_results.append(("✅", "Yahoo Finance", "Conectividade OK"))
                    else:
                        test_results.append(("❌", "Yahoo Finance", "Sem dados retornados"))
                except Exception as e:
                    test_results.append(("❌", "Yahoo Finance", f"Erro: {str(e)[:50]}..."))
                
                # Teste 2: Google Generative AI
                if GEMINI_AVAILABLE and st.session_state.model_configured:
                    try:
                        test_model = st.session_state.gemini_model
                        test_response = test_model.generate_content("Teste de conectividade")
                        test_results.append(("✅", "Google Gemini", "API funcionando"))
                    except Exception as e:
                        test_results.append(("❌", "Google Gemini", f"Erro: {str(e)[:50]}..."))
                else:
                    test_results.append(("⚠️", "Google Gemini", "Não configurado"))
                
                # Teste 3: TensorFlow
                try:
                    test_tensor = tf.constant([1, 2, 3])
                    test_results.append(("✅", "TensorFlow", f"Versão {tf.__version__}"))
                except Exception as e:
                    test_results.append(("❌", "TensorFlow", f"Erro: {str(e)[:50]}..."))
                
                # Teste 4: Bibliotecas de TA
                try:
                    test_data_ta = pd.Series([1, 2, 3, 4, 5])
                    test_sma = ta.trend.sma_indicator(test_data_ta, window=3)
                    test_results.append(("✅", "TA-Lib", "Indicadores funcionando"))
                except Exception as e:
                    test_results.append(("❌", "TA-Lib", f"Erro: {str(e)[:50]}..."))
            
            # Mostrar resultados
            st.markdown("**📋 Resultados dos Testes:**")
            for status, component, message in test_results:
                st.markdown(f"{status} **{component}:** {message}")
        
        # Informações do sistema
        st.markdown("#### 💻 Informações do Sistema")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📚 Versões das Bibliotecas**")
            try:
                import sys
                st.markdown(f"""
                - **Python:** {sys.version.split()[0]}
                - **Streamlit:** {st.__version__}
                - **TensorFlow:** {tf.__version__}
                - **Pandas:** {pd.__version__}
                - **NumPy:** {np.__version__}
                """)
            except:
                st.markdown("Erro ao obter versões")
        
        with col2:
            st.markdown("**🔧 Configurações**")
            st.markdown(f"""
            - **Cache ativo:** {'✅' if st.cache_data else '❌'}
            - **Modelos em cache:** {len(st.session_state.model_cache)}
            - **IA configurada:** {'✅' if st.session_state.model_configured else '❌'}
            - **Insights gerados:** {len(st.session_state.insights_data)}
            - **Modo verboso:** {'✅' if st.session_state.verbose_mode else '❌'}
            """)
        
        # Logs de carregamento
        if st.session_state.loading_logs:
            st.markdown("#### 📋 Logs de Carregamento")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Últimos {len(st.session_state.loading_logs)} eventos:**")
            with col2:
                if st.button("🗑️ Limpar Logs", use_container_width=True):
                    st.session_state.loading_logs = []
                    st.rerun()
            
            # Mostrar logs em container com scroll
            with st.container():
                for log_type, log_msg in reversed(st.session_state.loading_logs[-20:]):  # Últimos 20 logs
                    if log_type == "success":
                        st.success(log_msg)
                    elif log_type == "warning":
                        st.warning(log_msg)
                    elif log_type == "error":
                        st.error(log_msg)
                    else:
                        st.info(log_msg)
        else:
            st.markdown("#### 📋 Logs de Carregamento")
            st.info("Nenhum log disponível. Recarregue os dados para ver os logs.")
        
        # Botões de ação
        st.markdown("#### 🛠️ Ações de Manutenção")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Recarregar Dados", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache de dados limpo!")
                st.rerun()
        
        with col2:
            if st.button("🧠 Limpar Cache IA", use_container_width=True):
                st.session_state.insights_data = {}
                st.session_state.model_cache = {}
                st.success("Cache de IA limpo!")
        
        with col3:
            if st.button("📋 Limpar Logs", use_container_width=True):
                st.session_state.loading_logs = []
                st.success("Logs limpos!")
                st.rerun()
        
        with col4:
            if st.button("🔧 Reset Completo", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.cache_data.clear()
                st.success("Sistema resetado!")
                st.rerun()

# Disclaimer e rodapé
st.markdown("---")

# Warnings importantes
st.markdown("""
### ⚠️ Aviso Legal Importante

**Este sistema é apenas para fins educacionais e de demonstração.**

- 📊 As previsões são baseadas em análise técnica e machine learning
- 💰 **NÃO constitui recomendação de investimento**
- 📈 Desempenho passado não garante resultados futuros
- 🎯 Sempre consulte profissionais qualificados antes de investir
- 💸 Invista apenas o que pode perder
""")

# Rodapé informativo
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 15px; margin-top: 2rem;">
    <h3>🚀 StockAI Predictor v2.0</h3>
    <p><strong>Desenvolvido com Streamlit • TensorFlow • Google Generative AI</strong></p>
    <p><small>Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')}</small></p>
    <br>
    <p>📊 Dados: Yahoo Finance | 🤖 IA: Google Gemini | 📈 Indicadores: TA-Lib</p>
</div>
""", unsafe_allow_html=True)

# Executar apenas se for script principal
if __name__ == "__main__":
    pass