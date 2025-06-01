import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Conv1D, MaxPooling1D, Flatten, GRU, Bidirectional, Attention, MultiHeadAttention, LayerNormalization, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import ta  # Technical Analysis library
import joblib
import os
from datetime import datetime, timedelta
import warnings
import gc  # Garbage collection for memory management
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Configure TensorFlow for better memory usage
try:
    # For GPU users - limit memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"✅ GPU configurado: {len(gpus)} GPU(s) disponível(is)")
    else:
        logger.info("💻 Usando CPU para treinamento")
except Exception as e:
    logger.warning(f"⚠️ Configuração de GPU falhou: {e}")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# === Configuração Aprimorada ===
TICKERS = {
    'PETR4.SA': 'Petrobras',
    # 'VALE3.SA': 'Vale',
    # 'ITUB4.SA': 'Itaú Unibanco',
    # 'BBDC4.SA': 'Bradesco',
    # 'ABEV3.SA': 'Ambev',
    # 'WEGE3.SA': 'WEG',
    # 'MGLU3.SA': 'Magazine Luiza',
    # 'RENT3.SA': 'Localiza',
    # 'BPAC11.SA': 'BTG Pactual',
    # 'PRIO3.SA': 'PetroRio'
}

# Diretórios
os.makedirs('models', exist_ok=True)
os.makedirs('scalers', exist_ok=True)
os.makedirs('metrics', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# === MELHORIA 1: Função para adicionar features mais relevantes ===
def adicionar_features_otimizadas(df):
    """Adiciona features otimizadas focadas em padrões de preço"""
    df = df.copy()
    
    # Garantir que temos Series e não DataFrames
    close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
    high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
    low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']
    volume = df['Volume'].squeeze() if isinstance(df['Volume'], pd.DataFrame) else df['Volume']
    
    # === FEATURES PRINCIPAIS (mais relevantes para previsão) ===
    
    # 1. Retornos (mais fáceis de prever que preços absolutos)
    df['Return_1'] = close.pct_change(1)
    df['Return_2'] = close.pct_change(2)
    df['Return_5'] = close.pct_change(5)
    df['Return_10'] = close.pct_change(10)
    df['Return_20'] = close.pct_change(20)
    
    # 2. Médias móveis simples (apenas as mais importantes)
    df['SMA_5'] = ta.trend.sma_indicator(close, window=5)
    df['SMA_10'] = ta.trend.sma_indicator(close, window=10)
    df['SMA_20'] = ta.trend.sma_indicator(close, window=20)
    df['SMA_50'] = ta.trend.sma_indicator(close, window=50)
    
    # 3. Razões de preço para médias móveis (normalizadas)
    df['Price_to_SMA5'] = close / df['SMA_5'] - 1
    df['Price_to_SMA20'] = close / df['SMA_20'] - 1
    df['Price_to_SMA50'] = close / df['SMA_50'] - 1
    
    # 4. RSI (apenas um período)
    df['RSI_14'] = ta.momentum.rsi(close, window=14) / 100  # Normalizado entre 0 e 1
    
    # 5. MACD normalizado
    macd = ta.trend.MACD(close)
    df['MACD_diff_norm'] = macd.macd_diff() / close  # Normalizado pelo preço
    
    # 6. Bollinger Bands (apenas um período)
    bb = ta.volatility.BollingerBands(close, window=20)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    df['BB_position'] = (close - bb_lower) / (bb_upper - bb_lower)  # Posição relativa
    
    # 7. Volume normalizado
    df['Volume_ratio'] = volume / volume.rolling(20).mean()
    
    # 8. Volatilidade
    df['Volatility_5'] = close.pct_change().rolling(5).std()
    df['Volatility_20'] = close.pct_change().rolling(20).std()
    
    # 9. High-Low spread (volatilidade intraday)
    df['HL_spread'] = (high - low) / close
    
    # 10. Momentum
    df['Momentum_5'] = close / close.shift(5) - 1
    df['Momentum_10'] = close / close.shift(10) - 1
    
    # === MELHORIA: Adicionar o target como retorno futuro ===
    # Em vez de prever o preço absoluto, vamos prever o retorno
    df['Target_Return'] = close.shift(-1) / close - 1  # Retorno do próximo dia
    
    # Remover NaN
    df.dropna(inplace=True)
    
    return df

# === MELHORIA 2: Normalização melhorada ===
class ImprovedScaler:
    """Scaler customizado que normaliza cada feature independentemente"""
    def __init__(self, feature_range=(-1, 1)):
        self.scalers = {}
        self.feature_range = feature_range
        
    def fit_transform(self, X, feature_names):
        X_scaled = np.zeros_like(X)
        
        for i, feature in enumerate(feature_names):
            # Usar RobustScaler para features com outliers (como retornos)
            if 'Return' in feature or 'Momentum' in feature:
                scaler = RobustScaler()
            # Usar MinMaxScaler para features já limitadas (como RSI, BB_position)
            elif 'RSI' in feature or 'BB_position' in feature:
                scaler = MinMaxScaler(feature_range=self.feature_range)
            # Usar StandardScaler para o resto
            else:
                scaler = StandardScaler()
            
            X_scaled[:, i] = scaler.fit_transform(X[:, i].reshape(-1, 1)).ravel()
            self.scalers[feature] = scaler
            
        return X_scaled
    
    def transform(self, X, feature_names):
        X_scaled = np.zeros_like(X)
        
        for i, feature in enumerate(feature_names):
            X_scaled[:, i] = self.scalers[feature].transform(X[:, i].reshape(-1, 1)).ravel()
            
        return X_scaled
    
    def inverse_transform_target(self, y, close_prices):
        """Converte retornos previstos de volta para preços"""
        # y contém retornos previstos, close_prices são os últimos preços conhecidos
        return close_prices * (1 + y)

# === MELHORIA 3: Arquitetura de modelo mais robusta ===
def criar_modelo_robusto(input_shape, learning_rate=0.001):
    """Cria um modelo LSTM mais robusto e simples"""
    model = Sequential([
        # Primeira camada LSTM com menos unidades
        LSTM(64, return_sequences=True, input_shape=input_shape,
             kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
        Dropout(0.3),
        BatchNormalization(),
        
        # Segunda camada LSTM
        LSTM(32, return_sequences=False,
             kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
        Dropout(0.3),
        BatchNormalization(),
        
        # Camadas densas menores
        Dense(16, activation='relu'),
        Dropout(0.2),
        
        # Output - prevendo retorno (não preço)
        Dense(1, activation='tanh')  # tanh para retornos limitados
    ])
    
    # Optimizer com learning rate adaptativo
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)  # Gradient clipping
    
    model.compile(
        optimizer=optimizer,
        loss='mse',  # MSE para retornos
        metrics=['mae']
    )
    
    return model

# === MELHORIA 4: Criar conjunto de modelos (ensemble) ===
def criar_ensemble_modelos(input_shape, n_models=3):
    """Cria múltiplos modelos com diferentes arquiteturas"""
    models = []
    
    # Modelo 1: LSTM simples
    model1 = criar_modelo_robusto(input_shape, learning_rate=0.001)
    models.append(('lstm_simple', model1))
    
    # Modelo 2: GRU
    model2 = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        GRU(32, return_sequences=False),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='tanh')
    ])
    model2.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    models.append(('gru', model2))
    
    # Modelo 3: CNN-LSTM híbrido
    model3 = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='tanh')
    ])
    model3.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    models.append(('cnn_lstm', model3))
    
    return models

# === MELHORIA 5: Walk-forward validation ===
def walk_forward_validation(X, y, model_fn, n_splits=5, test_size=60):
    """Implementa walk-forward validation para séries temporais"""
    scores = []
    
    step = len(X) // n_splits
    
    for i in range(n_splits):
        # Definir janela de treino e teste
        test_end = len(X) - (i * step)
        test_start = test_end - test_size
        train_end = test_start
        
        if train_end < 100:  # Mínimo de dados para treino
            break
            
        # Dividir dados
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        
        # Criar e treinar modelo
        model = model_fn()
        
        # Callbacks mais conservadores
        callbacks = [
            EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.00001)
        ]
        
        # Treinar
        model.fit(X_train, y_train, epochs=50, batch_size=32, 
                 callbacks=callbacks, verbose=0)
        
        # Avaliar
        y_pred = model.predict(X_test, verbose=0)
        score = mean_squared_error(y_test, y_pred)
        scores.append(score)
        
        # Limpar memória
        del model
        tf.keras.backend.clear_session()
        
    return np.mean(scores), np.std(scores)

# === Função de treinamento principal melhorada ===
def treinar_modelo_ticker_melhorado(ticker, nome_ticker):
    """Versão melhorada do treinamento com foco em robustez"""
    print(f"\n{'='*80}")
    print(f"🚀 Treinando modelo melhorado para {nome_ticker} ({ticker})")
    print(f"{'='*80}")
    
    try:
        # 1. Coletar dados
        print("📊 Coletando dados históricos...")
        fim = datetime.now()
        inicio = fim - timedelta(days=365*10)  # 10 anos
        
        dados = yf.download(ticker, start=inicio, end=fim, progress=False)
        
        if len(dados) < 500:
            print(f"❌ Dados insuficientes para {ticker}")
            return None
            
        print(f"✅ {len(dados)} dias de dados coletados")
        
        # 2. Adicionar features otimizadas
        print("📈 Calculando features otimizadas...")
        dados = adicionar_features_otimizadas(dados)
        
        if len(dados) < 300:
            print(f"❌ Dados insuficientes após calcular features")
            return None
            
        # 3. Selecionar features (excluir colunas originais e target)
        feature_columns = [col for col in dados.columns if col not in 
                          ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Target_Return']]
        
        print(f"📊 Total de features: {len(feature_columns)}")
        
        # 4. Preparar dados
        X = dados[feature_columns].values
        y = dados['Target_Return'].values  # Agora prevendo retornos
        close_prices = dados['Close'].values  # Guardar para conversão posterior
        
        # 5. Normalização melhorada
        print("🔄 Normalizando dados...")
        scaler = ImprovedScaler()
        X_scaled = scaler.fit_transform(X, feature_columns)
        
        # 6. Criar sequências temporais
        janela = 30  # Janela menor para capturar padrões mais recentes
        X_seq, y_seq, close_seq = [], [], []
        
        for i in range(janela, len(X_scaled) - 1):
            X_seq.append(X_scaled[i-janela:i])
            y_seq.append(y[i])  # Retorno do próximo dia
            close_seq.append(close_prices[i])  # Preço atual para conversão
            
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        close_seq = np.array(close_seq)
        
        # 7. Divisão temporal dos dados (70/15/15)
        n_train = int(0.7 * len(X_seq))
        n_val = int(0.15 * len(X_seq))
        
        X_train = X_seq[:n_train]
        y_train = y_seq[:n_train]
        close_train = close_seq[:n_train]
        
        X_val = X_seq[n_train:n_train+n_val]
        y_val = y_seq[n_train:n_train+n_val]
        close_val = close_seq[n_train:n_train+n_val]
        
        X_test = X_seq[n_train+n_val:]
        y_test = y_seq[n_train+n_val:]
        close_test = close_seq[n_train+n_val:]
        
        print(f"📊 Dados divididos: {len(X_train)} treino, {len(X_val)} validação, {len(X_test)} teste")
        
        # 8. Treinar ensemble de modelos
        print("🤖 Treinando ensemble de modelos...")
        ensemble_models = []
        ensemble_histories = []
        
        for model_name, model in criar_ensemble_modelos((X_train.shape[1], X_train.shape[2])):
            print(f"   Treinando {model_name}...")
            
            # Callbacks melhorados
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001),
                ModelCheckpoint(f'checkpoints/{ticker}_{model_name}_best.keras', 
                              monitor='val_loss', save_best_only=True)
            ]
            
            # Treinar com batch size menor para mais atualizações
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=16,  # Batch menor
                callbacks=callbacks,
                verbose=0
            )
            
            ensemble_models.append((model_name, model))
            ensemble_histories.append(history)
            
        # 9. Avaliar ensemble
        print("📊 Avaliando ensemble...")
        ensemble_predictions = []
        
        for model_name, model in ensemble_models:
            y_pred_returns = model.predict(X_test, verbose=0).ravel()
            # Converter retornos previstos para preços
            y_pred_prices = close_test * (1 + y_pred_returns)
            ensemble_predictions.append(y_pred_prices)
            
        # Média das previsões
        y_pred_ensemble = np.mean(ensemble_predictions, axis=0)
        y_test_prices = close_test * (1 + y_test)  # Converter retornos reais para preços
        
        # 10. Métricas finais
        rmse = np.sqrt(mean_squared_error(y_test_prices, y_pred_ensemble))
        mae = mean_absolute_error(y_test_prices, y_pred_ensemble)
        r2 = r2_score(y_test_prices, y_pred_ensemble)
        
        # MAPE com proteção contra divisão por zero
        mask = y_test_prices != 0
        mape = np.mean(np.abs((y_test_prices[mask] - y_pred_ensemble[mask]) / y_test_prices[mask])) * 100
        
        # Acurácia de direção
        if len(y_test_prices) > 1:
            direction_accuracy = np.mean(
                (y_test_prices[1:] > y_test_prices[:-1]) == 
                (y_pred_ensemble[1:] > y_pred_ensemble[:-1])
            ) * 100
        else:
            direction_accuracy = 0
            
        print(f"\n✅ Resultados do Ensemble:")
        print(f"   RMSE: R$ {rmse:.2f}")
        print(f"   MAE: R$ {mae:.2f}")
        print(f"   R²: {r2:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   Acurácia de direção: {direction_accuracy:.2f}%")
        
        # 11. Salvar melhor modelo individual baseado em validação
        best_model_idx = np.argmin([h.history['val_loss'][-1] for h in ensemble_histories])
        best_model_name, best_model = ensemble_models[best_model_idx]
        
        print(f"\n💾 Salvando melhor modelo individual: {best_model_name}")
        best_model.save(f'models/{ticker}_advanced_model.keras')
        
        # Salvar scaler e informações
        joblib.dump(scaler, f'scalers/{ticker}_advanced_scaler.pkl')
        
        # Salvar métricas e metadados
        metricas = {
            'ticker': ticker,
            'nome': nome_ticker,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'best_model': best_model_name,
            'data_treino': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'janela': janela,
            'features': feature_columns,
            'prediction_type': 'returns'  # Importante: estamos prevendo retornos
        }
        joblib.dump(metricas, f'metrics/{ticker}_advanced_metrics.pkl')
        
        # 12. Visualização melhorada
        plt.figure(figsize=(20, 12))
        
        # Plot 1: Curvas de aprendizado do melhor modelo
        plt.subplot(3, 2, 1)
        best_history = ensemble_histories[best_model_idx]
        plt.plot(best_history.history['loss'], label='Treino', linewidth=2)
        plt.plot(best_history.history['val_loss'], label='Validação', linewidth=2)
        plt.title(f'Curvas de Perda - {nome_ticker} ({best_model_name})', fontsize=12)
        plt.xlabel('Épocas')
        plt.ylabel('Loss (MSE dos retornos)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Escala log para melhor visualização
        
        # Plot 2: Comparação dos modelos do ensemble
        plt.subplot(3, 2, 2)
        for i, (model_name, _) in enumerate(ensemble_models):
            val_losses = ensemble_histories[i].history['val_loss']
            plt.plot(val_losses, label=model_name, linewidth=2)
        plt.title('Comparação dos Modelos do Ensemble', fontsize=12)
        plt.xlabel('Épocas')
        plt.ylabel('Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Plot 3: Previsões vs Real (últimos 60 dias)
        plt.subplot(3, 2, 3)
        n_display = min(60, len(y_test_prices))
        indices = range(n_display)
        plt.plot(indices, y_test_prices[-n_display:], 'o-', label='Real', linewidth=2, markersize=4)
        plt.plot(indices, y_pred_ensemble[-n_display:], 's-', label='Ensemble', linewidth=2, markersize=3)
        
        # Adicionar previsões individuais em alpha menor
        for i, (model_name, _) in enumerate(ensemble_models):
            plt.plot(indices, ensemble_predictions[i][-n_display:], '--', 
                    label=f'{model_name}', alpha=0.5, linewidth=1)
        
        plt.title(f'Previsões vs Real (Últimos {n_display} dias)', fontsize=12)
        plt.xlabel('Dias')
        plt.ylabel('Preço (R$)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Distribuição dos erros percentuais
        plt.subplot(3, 2, 4)
        erros_pct = ((y_test_prices - y_pred_ensemble) / y_test_prices) * 100
        plt.hist(erros_pct, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.title('Distribuição dos Erros Percentuais', fontsize=12)
        plt.xlabel('Erro (%)')
        plt.ylabel('Frequência')
        plt.grid(True, alpha=0.3)
        
        # Adicionar estatísticas
        mean_error = np.mean(erros_pct)
        std_error = np.std(erros_pct)
        plt.text(0.05, 0.95, f'Média: {mean_error:.2f}%\nDesvio: {std_error:.2f}%', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 5: Scatter plot com linha de tendência
        plt.subplot(3, 2, 5)
        plt.scatter(y_test_prices, y_pred_ensemble, alpha=0.5, s=20)
        
        # Linha de regressão
        z = np.polyfit(y_test_prices, y_pred_ensemble, 1)
        p = np.poly1d(z)
        plt.plot(y_test_prices, p(y_test_prices), "r--", linewidth=2, label=f'Tendência: y={z[0]:.2f}x+{z[1]:.2f}')
        
        # Linha perfeita
        min_val = min(y_test_prices.min(), y_pred_ensemble.min())
        max_val = max(y_test_prices.max(), y_pred_ensemble.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfeito')
        
        plt.title('Correlação Real vs Previsto', fontsize=12)
        plt.xlabel('Preço Real (R$)')
        plt.ylabel('Preço Previsto (R$)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Métricas e informações
        plt.subplot(3, 2, 6)
        plt.text(0.1, 0.9, f'Métricas Finais - {nome_ticker}', fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
        plt.text(0.1, 0.75, f'RMSE: R$ {rmse:.2f}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.65, f'MAE: R$ {mae:.2f}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.55, f'R²: {r2:.4f}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.45, f'MAPE: {mape:.2f}%', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.35, f'Acurácia Direção: {direction_accuracy:.2f}%', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.25, f'Melhor Modelo: {best_model_name}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.15, f'Janela: {janela} dias', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.05, f'Features: {len(feature_columns)}', fontsize=12, transform=plt.gca().transAxes)
        
        # Adicionar indicador de qualidade
        if r2 > 0.5:
            quality = "✅ Excelente"
            color = 'green'
        elif r2 > 0.3:
            quality = "⚠️ Bom"
            color = 'orange'
        else:
            quality = "❌ Precisa melhorar"
            color = 'red'
            
        plt.text(0.6, 0.5, quality, fontsize=16, fontweight='bold', 
                color=color, transform=plt.gca().transAxes)
        
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'models/{ticker}_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Limpar memória
        for _, model in ensemble_models:
            del model
        tf.keras.backend.clear_session()
        gc.collect()
        
        return metricas
        
    except Exception as e:
        print(f"❌ Erro ao treinar {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# === Função melhorada para fazer previsões ===
def fazer_previsao_melhorada(ticker, dias_futuros=5):
    """Faz previsões usando o modelo melhorado"""
    try:
        # Carregar dados recentes
        fim = datetime.now()
        inicio = fim - timedelta(days=365)
        dados = yf.download(ticker, start=inicio, end=fim, progress=False)
        
        # Preparar dados com as mesmas features
        dados = adicionar_features_otimizadas(dados)
        
        # Carregar modelo e metadados
        modelo = tf.keras.models.load_model(f'models/{ticker}_advanced_model.keras')
        scaler = joblib.load(f'scalers/{ticker}_advanced_scaler.pkl')
        metricas = joblib.load(f'metrics/{ticker}_advanced_metrics.pkl')
        
        # Preparar features
        feature_columns = metricas['features']
        janela = metricas['janela']
        
        X = dados[feature_columns].values
        X_scaled = scaler.transform(X, feature_columns)
        
        # Pegar último preço conhecido
        ultimo_preco = dados['Close'].iloc[-1]
        
        # Criar janela para previsão
        ultima_janela = X_scaled[-janela:].reshape(1, janela, -1)
        
        # Fazer previsões iterativas
        previsoes = []
        preco_atual = ultimo_preco
        
        for i in range(dias_futuros):
            # Prever retorno
            retorno_previsto = modelo.predict(ultima_janela, verbose=0)[0, 0]
            
            # Converter para preço
            preco_previsto = preco_atual * (1 + retorno_previsto)
            previsoes.append(preco_previsto)
            
            # Atualizar para próxima previsão
            # (Aqui simplificamos - em produção, você atualizaria todas as features)
            preco_atual = preco_previsto
            
        return previsoes, dados.index[-1]
        
    except Exception as e:
        print(f"Erro na previsão: {str(e)}")
        return None, None

def verificar_modelos_existentes():
    """Verifica se já existem modelos treinados e seus desempenhos"""
    modelos_existentes = {}
    
    for ticker in TICKERS.keys():
        modelo_path = f'models/{ticker}_advanced_model.keras'
        metrics_path = f'metrics/{ticker}_advanced_metrics.pkl'
        
        if os.path.exists(modelo_path) and os.path.exists(metrics_path):
            try:
                metrics = joblib.load(metrics_path)
                modelos_existentes[ticker] = {
                    'r2': metrics.get('r2', 0),
                    'mape': metrics.get('mape', 100),
                    'direction_accuracy': metrics.get('direction_accuracy', 0),
                    'data_treino': metrics.get('data_treino', 'Desconhecido')
                }
            except Exception as e:
                logger.warning(f"⚠️ Erro ao carregar métricas de {ticker}: {e}")
                
    return modelos_existentes

# === EXECUTAR TREINAMENTO ===
if __name__ == "__main__":
    print("🚀 INICIANDO TREINAMENTO MELHORADO DOS MODELOS")
    print(f"📅 Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Total de tickers: {len(TICKERS)}")
    print("\n⚡ Usando técnicas melhoradas de ML...")
    
    # Verificar modelos existentes
    print("\n🔍 Verificando modelos existentes...")
    modelos_existentes = verificar_modelos_existentes()
    
    if modelos_existentes:
        print(f"✅ Encontrados {len(modelos_existentes)} modelos existentes:")
        for ticker, metrics in modelos_existentes.items():
            print(f"   {ticker}: R² = {metrics['r2']:.4f}, Treino: {metrics['data_treino']}")
    
    # Treinar modelos
    resultados = []
    
    for ticker, nome in TICKERS.items():
        print(f"\n{'='*80}")
        resultado = treinar_modelo_ticker_melhorado(ticker, nome)
        
        if resultado:
            resultados.append(resultado)
            
        # Pausa entre tickers
        import time
        time.sleep(2)
    
    # Resumo final
    print("\n" + "="*80)
    print("📊 RESUMO DO TREINAMENTO")
    print("="*80)
    
    if resultados:
        print(f"\n✅ Modelos treinados: {len(resultados)}/{len(TICKERS)}")
        
        # Estatísticas
        avg_r2 = np.mean([r['r2'] for r in resultados])
        avg_direction = np.mean([r['direction_accuracy'] for r in resultados])
        
        print(f"\n📊 Estatísticas:")
        print(f"   R² médio: {avg_r2:.4f}")
        print(f"   Acurácia direção média: {avg_direction:.2f}%")
        
        # Exemplo de previsão
        if resultados:
            melhor = max(resultados, key=lambda x: x['r2'])
            print(f"\n🔮 Exemplo de previsão para {melhor['nome']}:")
            
            previsoes, ultima_data = fazer_previsao_melhorada(melhor['ticker'], 5)
            
            if previsoes:
                print(f"Última data: {ultima_data.strftime('%Y-%m-%d')}")
                for i, prev in enumerate(previsoes, 1):
                    data_prev = ultima_data + timedelta(days=i)
                    print(f"   {data_prev.strftime('%Y-%m-%d')}: R$ {prev:.2f}")
    
    print("\n✅ Processo concluído!")