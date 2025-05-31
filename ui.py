# ============================================
# CÉLULA 2: INTERFACE STREAMLIT COM IA GENERATIVA
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tensorflow.keras.models import load_model
import joblib
import ta
from datetime import datetime, timedelta
import threading
import queue
import os
import time
import google.generativeai as genai

# Configuração da página
st.set_page_config(
    page_title="StockAI Predictor 📈",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para design moderno e claro
st.markdown("""
<style>
    /* Tema claro moderno */
    .stApp {
        background-color: #ffffff;
        color: #2c3e50;
    }

    /* Cards customizados - tema claro */
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #dee2e6;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        color: #2c3e50;
    }

    /* Títulos estilizados */
    .dashboard-header {
        background: linear-gradient(90deg, #4CAF50 0%, #2196F3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Botões customizados */
    .stButton > button {
        background: linear-gradient(90deg, #4CAF50 0%, #2196F3 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }

    /* Métricas destacadas */
    .big-metric {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4CAF50;
    }

    /* Cards de insight - tema claro */
    .insight-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        color: #1565c0;
        border: 1px solid #90caf9;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }

    /* Ajustar texto da sidebar */
    .css-1d391kg .stMarkdown {
        color: #2c3e50;
    }

    /* Métricas do Streamlit */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 10px;
    }

    /* Animação de loading */
    .loading-animation {
        animation: pulse 1.5s ease-in-out infinite;
    }

    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }

    /* Ajustar cores dos warnings e erros */
    .stAlert > div {
        border-radius: 10px;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 1rem;
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

# Configurar Google API Key
GOOGLE_API_KEY = st.sidebar.text_input(
    "🔑 Google API Key",
    type="password",
    help="Insira sua chave da API do Google para usar IA generativa"
)

# Funções auxiliares
def extrair_valor_escalar(series_ou_df):
    """Extrai um valor escalar de uma Series ou DataFrame de forma robusta"""
    try:
        if isinstance(series_ou_df, pd.DataFrame):
            if series_ou_df.empty:
                return 0.0
            return float(series_ou_df.iloc[0, 0])
        elif isinstance(series_ou_df, pd.Series):
            if series_ou_df.empty:
                return 0.0
            return float(series_ou_df.iloc[0])
        elif isinstance(series_ou_df, (int, float, np.number)):
            return float(series_ou_df)
        elif hasattr(series_ou_df, 'values'):
            # Para arrays numpy ou outros tipos array-like
            return float(series_ou_df.values.flatten()[0])
        else:
            return float(series_ou_df)
    except (IndexError, TypeError, ValueError) as e:
        st.warning(f"⚠️ Erro ao extrair valor escalar: {e}. Retornando 0.0")
        return 0.0

def adicionar_indicadores_tecnicos(df):
    """Adiciona indicadores técnicos ao DataFrame com tratamento robusto de erro"""
    try:
        if df is None or df.empty:
            st.error("❌ DataFrame vazio ou None fornecido para indicadores técnicos")
            return None
        
        df = df.copy()
        
        # Verificar se as colunas necessárias existem
        required_columns = ['Close', 'Volume', 'High', 'Low']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"❌ Colunas necessárias ausentes para indicadores técnicos: {missing_columns}")
            return None
        
        # Garantir que os dados sejam Series 1D e tratar valores NaN
        # Corrigindo o warning de depreciação do fillna(method=)
        def extract_series(column_data):
            """Extrai uma Series de uma coluna que pode ser multi-dimensional"""
            if isinstance(column_data, pd.DataFrame):
                # Se for DataFrame, pegar a primeira coluna
                series_data = column_data.iloc[:, 0]
            else:
                series_data = column_data
            return series_data.squeeze()
        
        close_prices = extract_series(df['Close']).ffill().bfill()
        volume = extract_series(df['Volume']).fillna(0)
        high_prices = extract_series(df['High']).ffill().bfill()
        low_prices = extract_series(df['Low']).ffill().bfill()
        
        # Verificar se temos dados suficientes
        if len(close_prices) < 21:  # Precisamos de pelo menos 21 dados para SMA_21
            st.error(f"❌ Dados insuficientes para calcular indicadores. Necessário: 21, disponível: {len(close_prices)}")
            return None
        
        # Calcular indicadores com tratamento de erro individual
        try:
            # Médias Móveis
            df['SMA_7'] = ta.trend.sma_indicator(close_prices, window=7)
            df['SMA_21'] = ta.trend.sma_indicator(close_prices, window=21)
            df['EMA_9'] = ta.trend.ema_indicator(close_prices, window=9)
        except Exception as e:
            st.error(f"❌ Erro ao calcular médias móveis: {str(e)}")
            return None
        
        try:
            # RSI
            df['RSI'] = ta.momentum.rsi(close_prices, window=14)
        except Exception as e:
            st.error(f"❌ Erro ao calcular RSI: {str(e)}")
            return None
        
        try:
            # MACD
            macd = ta.trend.MACD(close_prices)
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
        except Exception as e:
            st.error(f"❌ Erro ao calcular MACD: {str(e)}")
            return None
        
        try:
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(close_prices)
            df['BB_upper'] = bollinger.bollinger_hband()
            df['BB_lower'] = bollinger.bollinger_lband()
            df['BB_width'] = df['BB_upper'] - df['BB_lower']
        except Exception as e:
            st.error(f"❌ Erro ao calcular Bollinger Bands: {str(e)}")
            return None
        
        try:
            # Volume indicators - CORREÇÃO PRINCIPAL
            volume_sma = ta.trend.sma_indicator(volume, window=10)
            
            # Garantir que volume_sma seja uma Series e tratar divisão por zero
            if isinstance(volume_sma, pd.DataFrame):
                volume_sma = volume_sma.iloc[:, 0]
            
            df['Volume_SMA'] = volume_sma
            
            # Calcular Volume_ratio de forma segura
            volume_ratio = volume / volume_sma.replace(0, np.nan)
            df['Volume_ratio'] = volume_ratio
            
        except Exception as e:
            st.error(f"❌ Erro ao calcular indicadores de volume: {str(e)}")
            return None
        
        try:
            # Price features
            df['High_Low_pct'] = (high_prices - low_prices) / close_prices * 100
            df['Price_change'] = close_prices.pct_change()
        except Exception as e:
            st.error(f"❌ Erro ao calcular features de preço: {str(e)}")
            return None
        
        # Remover NaN values
        initial_length = len(df)
        df.dropna(inplace=True)
        final_length = len(df)
        
        if final_length == 0:
            st.error("❌ Todos os dados foram removidos após calcular indicadores técnicos")
            return None
        
        if final_length < initial_length * 0.5:  # Se perdemos mais de 50% dos dados
            st.warning(f"⚠️ Muitos dados foram removidos após calcular indicadores: {initial_length} → {final_length}")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Erro geral ao calcular indicadores técnicos: {str(e)}")
        return None

@st.cache_data
def carregar_dados_ticker(ticker, periodo='1y'):
    """Carrega dados do ticker com cache e melhor tratamento de erro"""
    try:
        # Verificar se o ticker é válido
        if not ticker or not isinstance(ticker, str):
            st.error(f"❌ Ticker inválido: {ticker}")
            return None
        
        # Log para debug
        st.info(f"🔄 Baixando dados para {ticker} (período: {periodo})...")
        
        # Download dos dados com timeout
        dados = yf.download(
            ticker, 
            period=periodo, 
            progress=False,
            timeout=30,
            threads=True,
            auto_adjust=False
        )
        
        # Verificar se dados foram baixados
        if dados is None or dados.empty:
            st.error(f"❌ Nenhum dado encontrado para {ticker}. Verifique se o ticker está correto.")
            return None
        
        # Verificar estrutura dos dados
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in dados.columns]
        if missing_columns:
            st.error(f"❌ Colunas obrigatórias ausentes: {missing_columns}")
            return None
        
        # Verificar se há dados suficientes
        if len(dados) < 50:
            st.error(f"❌ Dados insuficientes para {ticker}. Apenas {len(dados)} registros encontrados. Mínimo: 50")
            return None
        
        st.success(f"✅ {len(dados)} registros baixados com sucesso para {ticker}")
        
        # Processar dados
        if not isinstance(dados.index, pd.DatetimeIndex):
            dados = dados.reset_index()
            if 'Date' in dados.columns:
                dados.set_index('Date', inplace=True)
            elif 'Datetime' in dados.columns:
                dados.set_index('Datetime', inplace=True)
            else:
                st.error("❌ Coluna de data não encontrada nos dados")
                return None
        
        # Adicionar indicadores técnicos
        st.info("📊 Calculando indicadores técnicos...")
        dados_com_indicadores = adicionar_indicadores_tecnicos(dados)
        
        # Verificar se indicadores foram calculados
        if dados_com_indicadores is None or dados_com_indicadores.empty:
            st.error("❌ Erro ao calcular indicadores técnicos")
            return None
        
        # Verificar se ainda temos dados suficientes após calcular indicadores
        if len(dados_com_indicadores) < 30:
            st.error(f"❌ Dados insuficientes após calcular indicadores: {len(dados_com_indicadores)} registros")
            return None
        
        st.success(f"✅ Indicadores técnicos calculados. {len(dados_com_indicadores)} registros finais")
        return dados_com_indicadores
        
    except Exception as e:
        error_msg = f"❌ Erro ao carregar dados para {ticker}: {str(e)}"
        st.error(error_msg)
        
        # Sugestões de solução baseadas no tipo de erro
        if "timeout" in str(e).lower():
            st.warning("💡 Dica: Problema de conectividade. Tente novamente em alguns segundos.")
        elif "delisted" in str(e).lower() or "not found" in str(e).lower():
            st.warning("💡 Dica: Ticker pode estar incorreto ou a ação pode ter sido deslistada.")
        elif "rate limit" in str(e).lower():
            st.warning("💡 Dica: Muitas requisições. Aguarde alguns minutos antes de tentar novamente.")
        else:
            st.warning("💡 Dica: Verifique sua conexão com a internet e tente novamente.")
        
        # Log detalhado do erro para debug
        st.error(f"Detalhes técnicos: {type(e).__name__}: {str(e)}")
        return None

def configurar_gemini(api_key):
    """Configura o modelo Gemini"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        return model
    except Exception as e:
        st.error(f"Erro ao configurar Gemini: {str(e)}")
        return None

def gerar_insight_async(model, prompt, insight_type):
    """Gera insight de forma assíncrona"""
    try:
        # Configuração de segurança mais permissiva
        safety_settings = [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH"
            }
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
    """Inicia geração de insights em threads separadas"""

    # Preparar dados para os prompts - CORREÇÃO: extrair valores escalares
    preco_atual = extrair_valor_escalar(dados['Close'].iloc[-1:])
    variacao_mes = ((extrair_valor_escalar(dados['Close'].iloc[-1:]) / extrair_valor_escalar(dados['Close'].iloc[-30:-29])) - 1) * 100 if len(dados) > 30 else 0
    rsi_atual = extrair_valor_escalar(dados['RSI'].iloc[-1:])
    volume_medio = extrair_valor_escalar(dados['Volume'].mean())

    prompts = {
        'analise_tecnica': f"""
        Como analista financeiro especializado, analise os seguintes indicadores técnicos da ação {ticker}:
        - Preço atual: R$ {preco_atual:.2f}
        - RSI: {rsi_atual:.2f}
        - Variação mensal: {variacao_mes:.2f}%
        - Volume médio: {volume_medio:.0f}

        Forneça uma análise técnica concisa em 3-4 linhas, destacando se a ação está sobrecomprada, sobrevendida ou em equilíbrio.
        Use emojis para tornar a leitura mais agradável.
        """,

        'tendencia': f"""
        Analise a tendência da ação {ticker} com base nos seguintes dados:
        - Preço atual: R$ {preco_atual:.2f}
        - Média móvel 7 dias: R$ {extrair_valor_escalar(dados['SMA_7'].iloc[-1:]):.2f}
        - Média móvel 21 dias: R$ {extrair_valor_escalar(dados['SMA_21'].iloc[-1:]):.2f}
        - MACD: {extrair_valor_escalar(dados['MACD'].iloc[-1:]):.4f}

        Identifique se a tendência é de alta 📈, baixa 📉 ou lateral ➡️.
        Justifique sua análise em 2-3 linhas de forma clara e objetiva.
        """,

        'risco': f"""
        Avalie o nível de risco da ação {ticker} considerando:
        - Volatilidade (desvio padrão): {extrair_valor_escalar(dados['Close'].pct_change().std() * 100):.2f}%
        - R² do modelo de previsão: {metricas.get('r2', 0):.4f}
        - Largura das Bandas de Bollinger: {extrair_valor_escalar(dados['BB_width'].iloc[-1:]):.2f}

        Classifique o risco como:
        🟢 Baixo (volatilidade < 2%)
        🟡 Médio (volatilidade 2-4%)
        🔴 Alto (volatilidade > 4%)

        Explique sua classificação em 2-3 linhas.
        """,

        'recomendacao': f"""
        Com base na análise da ação {ticker}:
        - Preço atual: R$ {preco_atual:.2f}
        - RSI: {rsi_atual:.2f}
        - Volume em relação à média: {extrair_valor_escalar(dados['Volume_ratio'].iloc[-1:]):.2f}x
        - Tendência das médias móveis: {"Alta" if extrair_valor_escalar(dados['SMA_7'].iloc[-1:]) > extrair_valor_escalar(dados['SMA_21'].iloc[-1:]) else "Baixa"}

        Forneça uma sugestão estratégica (não é recomendação de investimento):
        💰 Momento de acumulação
        ⏸️ Aguardar melhor momento
        📊 Realizar lucros

        Justifique em 3-4 linhas e sempre mencione que isso não é conselho financeiro.
        """
    }

    # Criar threads para cada insight
    threads = []
    for tipo, prompt in prompts.items():
        t = threading.Thread(target=gerar_insight_async, args=(model, prompt, tipo))
        t.start()
        threads.append(t)

def testar_conectividade_yahoo():
    """Testa a conectividade com o Yahoo Finance usando um ticker confiável"""
    try:
        # Usar um ticker americano muito estável para teste
        ticker_teste = "AAPL"
        st.info(f"🔄 Testando conectividade com Yahoo Finance usando {ticker_teste}...")
        
        dados_teste = yf.download(
            ticker_teste, 
            period="5d", 
            progress=False,
            timeout=10,
            auto_adjust=False
        )
        
        if dados_teste is not None and not dados_teste.empty and len(dados_teste) > 0:
            st.success("✅ Conectividade com Yahoo Finance OK!")
            return True
        else:
            st.error("❌ Yahoo Finance não retornou dados válidos")
            return False
            
    except Exception as e:
        st.error(f"❌ Erro de conectividade com Yahoo Finance: {str(e)}")
        return False

# Interface principal
st.markdown('<h1 class="dashboard-header">📈 StockAI Predictor - Previsão Inteligente de Ações</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configurações")

    # Seleção de ticker
    tickers_disponiveis = {
        'PETR4.SA': 'Petrobras',
        'VALE3.SA': 'Vale',
        'ITUB4.SA': 'Itaú Unibanco',
        'BBDC4.SA': 'Bradesco',
        'ABEV3.SA': 'Ambev',
        'WEGE3.SA': 'WEG',
        'MGLU3.SA': 'Magazine Luiza',
        'RENT3.SA': 'Localiza',
        'BPAC11.SA': 'BTG Pactual',
        'PRIO3.SA': 'PetroRio'
    }

    ticker_selecionado = st.selectbox(
        "🎯 Selecione a Ação",
        options=list(tickers_disponiveis.keys()),
        format_func=lambda x: f"{tickers_disponiveis[x]} ({x})"
    )

    periodo = st.select_slider(
        "📅 Período de Análise",
        options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
        value='1y'
    )

    st.markdown("---")

    # Informações do modelo
    if os.path.exists(f'models/{ticker_selecionado}_model.h5'):
        try:
            metricas = joblib.load(f'metrics/{ticker_selecionado}_metrics.pkl')
            st.markdown("### 🤖 Modelo Treinado")
            st.success(f"✅ Modelo disponível")
            st.info(f"📅 Treinado em: {metricas['data_treino']}")
            st.metric("R² Score", f"{metricas['r2']:.4f}")
            st.metric("RMSE", f"R$ {metricas['rmse']:.2f}")
        except:
            st.error("❌ Erro ao carregar métricas do modelo")
    else:
        st.error("❌ Modelo não encontrado")
        st.warning("Execute a célula de treinamento primeiro!")
    
    st.markdown("---")
    
    # Seção de diagnóstico
    st.markdown("### 🔧 Diagnóstico")
    if st.button("🌐 Testar Conectividade", use_container_width=True):
        testar_conectividade_yahoo()
    
    # Botão para limpar cache
    if st.button("🗑️ Limpar Cache", use_container_width=True):
        st.cache_data.clear()
        st.success("✅ Cache limpo! Atualize a página.")
    
    st.markdown("---")
    
    # Informações de ajuda
    with st.expander("❓ Problemas Comuns"):
        st.markdown("""
        **Não consegue carregar dados?**
        - Verifique sua conexão com a internet
        - Teste a conectividade usando o botão acima
        - Tente outro ticker ou período menor
        
        **Dados muito antigos?**
        - Limpe o cache usando o botão acima
        - Atualize a página (F5)
        
        **Erro de modelo?**
        - Execute primeiro o treinamento.py
        - Verifique se os arquivos foram salvos corretamente
        """)

# Configurar modelo Gemini se a API key foi fornecida
if GOOGLE_API_KEY and not st.session_state.model_configured:
    model = configurar_gemini(GOOGLE_API_KEY)
    if model:
        st.session_state.model_configured = True
        st.session_state.gemini_model = model

# Layout principal
if ticker_selecionado and os.path.exists(f'models/{ticker_selecionado}_model.h5'):

    # Carregar dados e modelo
    with st.spinner('Carregando dados...'):
        dados = carregar_dados_ticker(ticker_selecionado, periodo)
        try:
            # CORREÇÃO: Carregando modelo sem compilar para evitar erro de desserialização
            modelo = load_model(f'models/{ticker_selecionado}_model.h5', compile=False)
            scaler = joblib.load(f'scalers/{ticker_selecionado}_scaler.pkl')
            metricas = joblib.load(f'metrics/{ticker_selecionado}_metrics.pkl')
        except Exception as e:
            st.error(f"Erro ao carregar modelo: {str(e)}")
            st.stop()

    if dados is not None and len(dados) > 0:
        # Tabs principais
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔮 Previsão", "🤖 Insights IA", "📈 Análise Técnica"])

        with tab1:
            # Métricas principais
            col1, col2, col3, col4 = st.columns(4)

            # CORREÇÃO: Extrair valores escalares de forma segura
            preco_atual = extrair_valor_escalar(dados['Close'].iloc[-1:])
            preco_anterior = extrair_valor_escalar(dados['Close'].iloc[-2:-1])
            variacao_diaria = ((preco_atual - preco_anterior) / preco_anterior) * 100

            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("💰 Preço Atual", f"R$ {preco_atual:.2f}",
                         f"{variacao_diaria:+.2f}%",
                         delta_color="normal" if variacao_diaria >= 0 else "inverse")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                volume_atual = extrair_valor_escalar(dados['Volume'].iloc[-1:])
                volume_medio = extrair_valor_escalar(dados['Volume'].mean())
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("📊 Volume", f"{volume_atual:,.0f}",
                         f"{((volume_atual/volume_medio - 1) * 100):+.1f}% vs média")
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                rsi = extrair_valor_escalar(dados['RSI'].iloc[-1:])
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("📈 RSI", f"{rsi:.2f}",
                         "Sobrecomprado" if rsi > 70 else "Sobrevendido" if rsi < 30 else "Neutro")
                st.markdown('</div>', unsafe_allow_html=True)

            with col4:
                volatilidade = extrair_valor_escalar(dados['Close'].pct_change().std() * 100)
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("📊 Volatilidade", f"{volatilidade:.2f}%",
                         "Alta" if volatilidade > 3 else "Baixa")
                st.markdown('</div>', unsafe_allow_html=True)

            # Gráfico principal
            st.markdown("### 📈 Gráfico de Preços e Volume")

            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=("Preço e Médias Móveis", "Volume")
            )

            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=dados.index,
                    open=dados['Open'],
                    high=dados['High'],
                    low=dados['Low'],
                    close=dados['Close'],
                    name="Preço",
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350'
                ),
                row=1, col=1
            )

            # Médias móveis
            fig.add_trace(
                go.Scatter(
                    x=dados.index,
                    y=dados['SMA_7'],
                    name="SMA 7",
                    line=dict(color='#2196F3', width=2)
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=dados.index,
                    y=dados['SMA_21'],
                    name="SMA 21",
                    line=dict(color='#FFC107', width=2)
                ),
                row=1, col=1
            )

            # Volume
            colors = []
            for i in range(len(dados)):
                close_val = extrair_valor_escalar(dados['Close'].iloc[i:i+1])
                open_val = extrair_valor_escalar(dados['Open'].iloc[i:i+1])
                colors.append('#26a69a' if close_val >= open_val else '#ef5350')

            fig.add_trace(
                go.Bar(
                    x=dados.index,
                    y=dados['Volume'],
                    name="Volume",
                    marker_color=colors
                ),
                row=2, col=1
            )

            fig.update_layout(
                template="plotly_white",
                height=600,
                showlegend=True,
                xaxis_rangeslider_visible=False,
                margin=dict(l=0, r=0, t=30, b=0)
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("### 🔮 Previsão para o Próximo Dia")

            # Preparar dados para previsão
            features = metricas['features']
            janela = metricas['janela']

            # Pegar últimos dados
            if len(dados) >= janela:
                ultimos_dados = dados[features].iloc[-janela:].values
                ultimos_dados_norm = scaler.transform(ultimos_dados)

                # Fazer previsão
                X_pred = ultimos_dados_norm.reshape(1, janela, len(features))
                previsao_norm = modelo.predict(X_pred, verbose=0)

                # Desnormalizar
                previsao_completa = np.zeros((1, len(features)))
                previsao_completa[0, 0] = previsao_norm[0, 0]
                previsao_real = scaler.inverse_transform(previsao_completa)[0, 0]

                # Calcular variação prevista
                variacao_prevista = ((previsao_real - preco_atual) / preco_atual) * 100

                # Display da previsão
                col1, col2, col3 = st.columns([1, 2, 1])

                with col2:
                    st.markdown('<div class="metric-card" style="text-align: center;">', unsafe_allow_html=True)
                    st.markdown(f'<h2 class="big-metric">R$ {previsao_real:.2f}</h2>', unsafe_allow_html=True)
                    st.markdown(f"<h4>Variação Prevista: {variacao_prevista:+.2f}%</h4>", unsafe_allow_html=True)

                    if variacao_prevista > 0:
                        st.success(f"📈 Tendência de ALTA")
                    else:
                        st.error(f"📉 Tendência de BAIXA")

                    st.markdown('</div>', unsafe_allow_html=True)

                # Confiança do modelo
                st.markdown("### 📊 Confiança da Previsão")

                col1, col2, col3 = st.columns(3)

                with col1:
                    confianca_r2 = metricas['r2'] * 100
                    st.metric("🎯 Precisão Histórica", f"{confianca_r2:.1f}%")

                with col2:
                    st.metric("📉 Erro Médio (MAE)", f"R$ {metricas['mae']:.2f}")

                with col3:
                    risco_score = min(100, (metricas['rmse'] / extrair_valor_escalar(preco_atual)) * 100)
                    st.metric("⚠️ Nível de Risco", f"{risco_score:.1f}%")

                # Gráfico de previsão
                st.markdown("### 📈 Visualização da Previsão")

                # Últimos 30 dias + previsão
                ultimos_30_dias = dados['Close'].iloc[-30:].copy()
                datas = pd.date_range(start=ultimos_30_dias.index[-1] + timedelta(days=1), periods=1)

                fig_prev = go.Figure()

                # Histórico
                fig_prev.add_trace(go.Scatter(
                    x=ultimos_30_dias.index,
                    y=ultimos_30_dias.values,
                    mode='lines',
                    name='Histórico',
                    line=dict(color='#2196F3', width=3)
                ))

                # Previsão
                fig_prev.add_trace(go.Scatter(
                    x=[ultimos_30_dias.index[-1], datas[0]],
                    y=[extrair_valor_escalar(preco_atual), previsao_real],
                    mode='lines+markers',
                    name='Previsão',
                    line=dict(color='#4CAF50', width=3, dash='dash'),
                    marker=dict(size=10)
                ))

                fig_prev.update_layout(
                    template="plotly_white",
                    height=400,
                    title="Últimos 30 dias + Previsão",
                    xaxis_title="Data",
                    yaxis_title="Preço (R$)",
                    showlegend=True
                )

                st.plotly_chart(fig_prev, use_container_width=True)
            else:
                st.error(f"Dados insuficientes. Necessário pelo menos {janela} dias de histórico.")

            # Disclaimer
            st.warning("⚠️ **Aviso Legal**: Esta previsão é baseada em análise técnica e machine learning. Não constitui recomendação de investimento. Sempre consulte um profissional qualificado antes de tomar decisões de investimento.")

        with tab3:
            st.markdown("### 🤖 Insights de IA Generativa")

            if GOOGLE_API_KEY and st.session_state.model_configured:
                # Botão para gerar insights
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🚀 Gerar Insights com IA", use_container_width=True):
                        st.session_state.insights_data = {}
                        model = st.session_state.gemini_model
                        iniciar_geracao_insights(model, ticker_selecionado, dados, metricas)

                # Coletar insights da fila
                insights_recebidos = {}
                while not st.session_state.insights_queue.empty():
                    tipo, conteudo = st.session_state.insights_queue.get()
                    insights_recebidos[tipo] = conteudo

                # Atualizar insights no estado
                st.session_state.insights_data.update(insights_recebidos)

                # Display dos insights
                col1, col2 = st.columns(2)

                with col1:
                    if 'analise_tecnica' in st.session_state.insights_data:
                        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                        st.markdown("#### 📊 Análise Técnica")
                        st.markdown(st.session_state.insights_data['analise_tecnica'])
                        st.markdown('</div>', unsafe_allow_html=True)

                    if 'risco' in st.session_state.insights_data:
                        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                        st.markdown("#### ⚠️ Avaliação de Risco")
                        st.markdown(st.session_state.insights_data['risco'])
                        st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    if 'tendencia' in st.session_state.insights_data:
                        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                        st.markdown("#### 📈 Tendência")
                        st.markdown(st.session_state.insights_data['tendencia'])
                        st.markdown('</div>', unsafe_allow_html=True)

                    if 'recomendacao' in st.session_state.insights_data:
                        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                        st.markdown("#### 💡 Recomendação")
                        st.markdown(st.session_state.insights_data['recomendacao'])
                        st.markdown('</div>', unsafe_allow_html=True)

                # Se há insights sendo gerados, mostrar spinner
                if st.session_state.insights_data and len(st.session_state.insights_data) < 4:
                    with st.spinner("🔄 Gerando mais insights..."):
                        time.sleep(2)
                        st.rerun()

            else:
                st.warning("🔑 Por favor, insira sua Google API Key na barra lateral para ativar os insights de IA.")
                st.info("Você pode obter uma chave gratuitamente em: https://makersuite.google.com/app/apikey")

        with tab4:
            st.markdown("### 📈 Análise Técnica Detalhada")

            # Indicadores em colunas
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### 📊 Médias Móveis")
                st.metric("SMA 7", f"R$ {extrair_valor_escalar(dados['SMA_7'].iloc[-1:]):.2f}")
                st.metric("SMA 21", f"R$ {extrair_valor_escalar(dados['SMA_21'].iloc[-1:]):.2f}")
                st.metric("EMA 9", f"R$ {extrair_valor_escalar(dados['EMA_9'].iloc[-1:]):.2f}")

            with col2:
                st.markdown("#### 📈 Momentum")
                st.metric("RSI (14)", f"{extrair_valor_escalar(dados['RSI'].iloc[-1:]):.2f}")
                st.metric("MACD", f"{extrair_valor_escalar(dados['MACD'].iloc[-1:]):.4f}")
                st.metric("MACD Signal", f"{extrair_valor_escalar(dados['MACD_signal'].iloc[-1:]):.4f}")

            with col3:
                st.markdown("#### 📉 Volatilidade")
                st.metric("Bollinger Superior", f"R$ {extrair_valor_escalar(dados['BB_upper'].iloc[-1:]):.2f}")
                st.metric("Bollinger Inferior", f"R$ {extrair_valor_escalar(dados['BB_lower'].iloc[-1:]):.2f}")
                st.metric("Largura BB", f"{extrair_valor_escalar(dados['BB_width'].iloc[-1:]):.2f}")

            # Gráficos de indicadores
            st.markdown("### 📊 Visualização de Indicadores")

            # RSI
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=dados.index,
                y=dados['RSI'],
                name='RSI',
                line=dict(color='#2196F3', width=2)
            ))

            # Linhas de referência
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Sobrecomprado")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Sobrevendido")

            fig_rsi.update_layout(
                template="plotly_white",
                height=300,
                title="RSI (14)",
                yaxis_title="RSI",
                xaxis_title="Data"
            )

            st.plotly_chart(fig_rsi, use_container_width=True)

            # MACD
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(
                x=dados.index,
                y=dados['MACD'],
                name='MACD',
                line=dict(color='#4CAF50', width=2)
            ))

            fig_macd.add_trace(go.Scatter(
                x=dados.index,
                y=dados['MACD_signal'],
                name='Signal',
                line=dict(color='#FFC107', width=2)
            ))

            # Histograma MACD
            macd_hist = dados['MACD'] - dados['MACD_signal']
            colors = []
            for val in macd_hist:
                val_escalar = extrair_valor_escalar(val) if hasattr(val, 'iloc') else float(val)
                colors.append('#26a69a' if val_escalar >= 0 else '#ef5350')

            fig_macd.add_trace(go.Bar(
                x=dados.index,
                y=macd_hist,
                name='Histograma',
                marker_color=colors
            ))

            fig_macd.update_layout(
                template="plotly_white",
                height=300,
                title="MACD",
                yaxis_title="Valor",
                xaxis_title="Data"
            )

            st.plotly_chart(fig_macd, use_container_width=True)
    else:
        st.error("❌ Não foi possível carregar os dados da ação selecionada.")

# Rodapé
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888;">
        <p>Desenvolvido usando Streamlit e Google Generative AI</p>
        <p>StockAI Predictor v1.0 - Projeto de TCC</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Para executar no Colab, salve este código em app.py e execute:
# !streamlit run app.py --server.port 8501 &
# Em seguida, use ngrok para expor a porta 8501
