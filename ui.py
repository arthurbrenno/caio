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

# CSS customizado para design moderno
st.markdown("""
<style>
    /* Tema escuro moderno */
    .stApp {
        background-color: #0e1117;
    }

    /* Cards customizados */
    .metric-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
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
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
    }

    /* Métricas destacadas */
    .big-metric {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4CAF50;
    }

    /* Cards de insight */
    .insight-card {
        background: linear-gradient(135deg, #1a237e 0%, #3949ab 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
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
def adicionar_indicadores_tecnicos(df):
    """Adiciona indicadores técnicos ao DataFrame"""
    df = df.copy()

    # Garantir que os dados sejam Series 1D
    close_prices = df['Close'].squeeze()
    volume = df['Volume'].squeeze()
    high_prices = df['High'].squeeze()
    low_prices = df['Low'].squeeze()

    # Médias Móveis
    df['SMA_7'] = ta.trend.sma_indicator(close_prices, window=7)
    df['SMA_21'] = ta.trend.sma_indicator(close_prices, window=21)
    df['EMA_9'] = ta.trend.ema_indicator(close_prices, window=9)

    # RSI
    df['RSI'] = ta.momentum.rsi(close_prices, window=14)

    # MACD
    macd = ta.trend.MACD(close_prices)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close_prices)
    df['BB_upper'] = bollinger.bollinger_hband()
    df['BB_lower'] = bollinger.bollinger_lband()
    df['BB_width'] = df['BB_upper'] - df['BB_lower']

    # Volume indicators
    df['Volume_SMA'] = ta.trend.sma_indicator(volume, window=10)
    df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']

    # Price features
    df['High_Low_pct'] = (high_prices - low_prices) / close_prices * 100
    df['Price_change'] = close_prices.pct_change()

    df.dropna(inplace=True)
    return df

@st.cache_data
def carregar_dados_ticker(ticker, periodo='1y'):
    """Carrega dados do ticker com cache"""
    try:
        dados = yf.download(ticker, period=periodo, progress=False)
        dados = dados.reset_index()
        dados.set_index('Date', inplace=True)
        return adicionar_indicadores_tecnicos(dados)
    except:
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

    # Preparar dados para os prompts
    preco_atual = dados['Close'].iloc[-1]
    variacao_mes = ((dados['Close'].iloc[-1] / dados['Close'].iloc[-30]) - 1) * 100 if len(dados) > 30 else 0
    rsi_atual = dados['RSI'].iloc[-1]
    volume_medio = dados['Volume'].mean()

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
        - Média móvel 7 dias: R$ {dados['SMA_7'].iloc[-1]:.2f}
        - Média móvel 21 dias: R$ {dados['SMA_21'].iloc[-1]:.2f}
        - MACD: {dados['MACD'].iloc[-1]:.4f}

        Identifique se a tendência é de alta 📈, baixa 📉 ou lateral ➡️.
        Justifique sua análise em 2-3 linhas de forma clara e objetiva.
        """,

        'risco': f"""
        Avalie o nível de risco da ação {ticker} considerando:
        - Volatilidade (desvio padrão): {dados['Close'].pct_change().std() * 100:.2f}%
        - R² do modelo de previsão: {metricas.get('r2', 0):.4f}
        - Largura das Bandas de Bollinger: {dados['BB_width'].iloc[-1]:.2f}

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
        - Volume em relação à média: {dados['Volume_ratio'].iloc[-1]:.2f}x
        - Tendência das médias móveis: {"Alta" if dados['SMA_7'].iloc[-1] > dados['SMA_21'].iloc[-1] else "Baixa"}

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
            modelo = load_model(f'models/{ticker_selecionado}_model.h5')
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

            preco_atual = dados['Close'].iloc[-1]
            preco_anterior = dados['Close'].iloc[-2]
            variacao_diaria = ((preco_atual - preco_anterior) / preco_anterior) * 100

            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("💰 Preço Atual", f"R$ {preco_atual:.2f}",
                         f"{variacao_diaria:+.2f}%",
                         delta_color="normal" if variacao_diaria >= 0 else "inverse")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                volume_atual = dados['Volume'].iloc[-1]
                volume_medio = dados['Volume'].mean()
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("📊 Volume", f"{volume_atual:,.0f}",
                         f"{((volume_atual/volume_medio - 1) * 100):+.1f}% vs média")
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                rsi = dados['RSI'].iloc[-1]
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("📈 RSI", f"{rsi:.2f}",
                         "Sobrecomprado" if rsi > 70 else "Sobrevendido" if rsi < 30 else "Neutro")
                st.markdown('</div>', unsafe_allow_html=True)

            with col4:
                volatilidade = dados['Close'].pct_change().std() * 100
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
            colors = ['#26a69a' if dados['Close'].iloc[i] >= dados['Open'].iloc[i]
                     else '#ef5350' for i in range(len(dados))]

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
                template="plotly_dark",
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
                    risco_score = min(100, (metricas['rmse'] / preco_atual) * 100)
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
                    y=[preco_atual, previsao_real],
                    mode='lines+markers',
                    name='Previsão',
                    line=dict(color='#4CAF50', width=3, dash='dash'),
                    marker=dict(size=10)
                ))

                fig_prev.update_layout(
                    template="plotly_dark",
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
                st.metric("SMA 7", f"R$ {dados['SMA_7'].iloc[-1]:.2f}")
                st.metric("SMA 21", f"R$ {dados['SMA_21'].iloc[-1]:.2f}")
                st.metric("EMA 9", f"R$ {dados['EMA_9'].iloc[-1]:.2f}")

            with col2:
                st.markdown("#### 📈 Momentum")
                st.metric("RSI (14)", f"{dados['RSI'].iloc[-1]:.2f}")
                st.metric("MACD", f"{dados['MACD'].iloc[-1]:.4f}")
                st.metric("MACD Signal", f"{dados['MACD_signal'].iloc[-1]:.4f}")

            with col3:
                st.markdown("#### 📉 Volatilidade")
                st.metric("Bollinger Superior", f"R$ {dados['BB_upper'].iloc[-1]:.2f}")
                st.metric("Bollinger Inferior", f"R$ {dados['BB_lower'].iloc[-1]:.2f}")
                st.metric("Largura BB", f"{dados['BB_width'].iloc[-1]:.2f}")

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
                template="plotly_dark",
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
            colors = ['#26a69a' if val >= 0 else '#ef5350' for val in macd_hist]

            fig_macd.add_trace(go.Bar(
                x=dados.index,
                y=macd_hist,
                name='Histograma',
                marker_color=colors
            ))

            fig_macd.update_layout(
                template="plotly_dark",
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
