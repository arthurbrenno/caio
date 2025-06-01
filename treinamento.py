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

# Diretórios
os.makedirs('models', exist_ok=True)
os.makedirs('scalers', exist_ok=True)
os.makedirs('metrics', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# === Função Aprimorada para Indicadores Técnicos ===
def adicionar_indicadores_tecnicos_avancados(df):
    """Adiciona indicadores técnicos avançados ao DataFrame"""
    df = df.copy()
    
    # Garantir que as colunas sejam Series
    close_prices = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
    volume_series = df['Volume'].squeeze() if isinstance(df['Volume'], pd.DataFrame) else df['Volume']
    high_prices = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
    low_prices = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']
    open_prices = df['Open'].squeeze() if isinstance(df['Open'], pd.DataFrame) else df['Open']
    
    # Médias Móveis (múltiplos períodos)
    for period in [5, 7, 9, 10, 20, 21, 50, 100, 200]:
        df[f'SMA_{period}'] = ta.trend.sma_indicator(close_prices, window=period)
        df[f'EMA_{period}'] = ta.trend.ema_indicator(close_prices, window=period)
    
    # RSI (múltiplos períodos)
    for period in [9, 14, 21]:
        df[f'RSI_{period}'] = ta.momentum.rsi(close_prices, window=period)
    
    # MACD
    macd_obj = ta.trend.MACD(close_prices)
    df['MACD'] = macd_obj.macd()
    df['MACD_signal'] = macd_obj.macd_signal()
    df['MACD_diff'] = macd_obj.macd_diff()
    
    # Bollinger Bands
    for period in [10, 20, 30]:
        bollinger_obj = ta.volatility.BollingerBands(close_prices, window=period)
        df[f'BB_upper_{period}'] = bollinger_obj.bollinger_hband()
        df[f'BB_lower_{period}'] = bollinger_obj.bollinger_lband()
        df[f'BB_width_{period}'] = df[f'BB_upper_{period}'] - df[f'BB_lower_{period}']
        df[f'BB_pctb_{period}'] = (close_prices - df[f'BB_lower_{period}']) / df[f'BB_width_{period}']
    
    # Stochastic Oscillator
    stoch_obj = ta.momentum.StochasticOscillator(high_prices, low_prices, close_prices)
    df['Stoch_K'] = stoch_obj.stoch()
    df['Stoch_D'] = stoch_obj.stoch_signal()
    
    # ATR (Average True Range) - Volatilidade
    for period in [7, 14, 21]:
        df[f'ATR_{period}'] = ta.volatility.average_true_range(high_prices, low_prices, close_prices, window=period)
    
    # ADX (Average Directional Index)
    adx_obj = ta.trend.ADXIndicator(high_prices, low_prices, close_prices)
    df['ADX'] = adx_obj.adx()
    df['ADX_pos'] = adx_obj.adx_pos()
    df['ADX_neg'] = adx_obj.adx_neg()
    
    # CCI (Commodity Channel Index)
    df['CCI'] = ta.trend.cci(high_prices, low_prices, close_prices)
    
    # Williams %R
    df['Williams_R'] = ta.momentum.williams_r(high_prices, low_prices, close_prices)
    
    # MFI (Money Flow Index)
    df['MFI'] = ta.volume.money_flow_index(high_prices, low_prices, close_prices, volume_series)
    
    # OBV (On Balance Volume)
    df['OBV'] = ta.volume.on_balance_volume(close_prices, volume_series)
    df['OBV_EMA'] = ta.trend.ema_indicator(df['OBV'], window=20)
    
    # Volume indicators
    df['Volume_SMA_10'] = ta.trend.sma_indicator(volume_series, window=10)
    df['Volume_SMA_20'] = ta.trend.sma_indicator(volume_series, window=20)
    df['Volume_ratio'] = volume_series / df['Volume_SMA_20']
    df['Force_Index'] = close_prices.diff() * volume_series
    
    # Price patterns
    df['High_Low_pct'] = (high_prices - low_prices) / close_prices * 100
    df['Close_Open_pct'] = (close_prices - open_prices) / open_prices * 100
    
    # Returns (múltiplos períodos)
    for period in [1, 2, 3, 5, 10, 20]:
        df[f'Return_{period}'] = close_prices.pct_change(period)
        df[f'Log_Return_{period}'] = np.log(close_prices / close_prices.shift(period))
    
    # Volatilidade histórica
    for period in [5, 10, 20, 30]:
        df[f'Volatility_{period}'] = close_prices.pct_change().rolling(period).std()
    
    # Pivot Points
    df['Pivot'] = (high_prices + low_prices + close_prices) / 3
    df['R1'] = 2 * df['Pivot'] - low_prices
    df['S1'] = 2 * df['Pivot'] - high_prices
    
    # Ichimoku Cloud
    period9_high = high_prices.rolling(9).max()
    period9_low = low_prices.rolling(9).min()
    df['Ichimoku_conv'] = (period9_high + period9_low) / 2
    
    period26_high = high_prices.rolling(26).max()
    period26_low = low_prices.rolling(26).min()
    df['Ichimoku_base'] = (period26_high + period26_low) / 2
    
    # Parabolic SAR
    psar_obj = ta.trend.PSARIndicator(high_prices, low_prices, close_prices)
    df['PSAR'] = psar_obj.psar()
    
    # Feature Engineering adicional
    # Razões entre médias móveis
    df['SMA_5_20_ratio'] = df['SMA_5'] / df['SMA_20']
    df['SMA_20_50_ratio'] = df['SMA_20'] / df['SMA_50']
    df['EMA_9_21_ratio'] = df['EMA_9'] / df['EMA_21']
    
    # Distância do preço às médias móveis
    df['Price_SMA20_distance'] = (close_prices - df['SMA_20']) / df['SMA_20'] * 100
    df['Price_SMA50_distance'] = (close_prices - df['SMA_50']) / df['SMA_50'] * 100
    
    # Contadores de tendência
    df['Days_above_SMA20'] = (close_prices > df['SMA_20']).rolling(20).sum()
    df['Days_above_SMA50'] = (close_prices > df['SMA_50']).rolling(50).sum()
    
    # Features de tempo
    df['DayOfWeek'] = df.index.dayofweek
    df['DayOfMonth'] = df.index.day
    df['MonthOfYear'] = df.index.month
    df['Quarter'] = df.index.quarter
    
    # Encoding cíclico para features temporais
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['MonthOfYear_sin'] = np.sin(2 * np.pi * df['MonthOfYear'] / 12)
    df['MonthOfYear_cos'] = np.cos(2 * np.pi * df['MonthOfYear'] / 12)
    
    # Remove NaN values
    df.dropna(inplace=True)
    
    return df

# === Data Augmentation para Séries Temporais ===
def augment_time_series(X, y, noise_factor=0.01, shift_range=5):
    """Aplica data augmentation em séries temporais"""
    augmented_X = []
    augmented_y = []
    
    # Original data
    augmented_X.append(X)
    augmented_y.append(y)
    
    # Adicionar ruído gaussiano
    noise = np.random.normal(0, noise_factor, X.shape)
    augmented_X.append(X + noise)
    augmented_y.append(y)
    
    # Time shifting
    for shift in range(1, shift_range + 1):
        if len(X) > shift:
            augmented_X.append(X[:-shift])
            augmented_y.append(y[:-shift])
    
    return np.concatenate(augmented_X), np.concatenate(augmented_y)

# === Função para criar janelas com mais contexto ===
def criar_janelas_avancadas(dados, janela=60, horizonte=5):
    """Cria janelas temporais com múltiplos horizontes de previsão"""
    X, y = [], []
    for i in range(len(dados) - janela - horizonte + 1):
        X.append(dados[i:i+janela])
        # Prevê múltiplos passos à frente
        y.append(dados[i+janela:i+janela+horizonte, 0])  # Prevê apenas Close
    return np.array(X), np.array(y)

# === Modelo LSTM com Attention Mechanism ===
def criar_modelo_attention_lstm(input_shape, output_steps=1):
    """Cria um modelo LSTM com mecanismo de atenção simplificado"""
    inputs = Input(shape=input_shape)
    
    # Primeira camada LSTM bidirecional
    lstm1 = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))(inputs)
    lstm1 = BatchNormalization()(lstm1)
    lstm1 = Dropout(0.3)(lstm1)
    
    # Segunda camada LSTM bidirecional
    lstm2 = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))(lstm1)
    lstm2 = BatchNormalization()(lstm2)
    lstm2 = Dropout(0.3)(lstm2)
    
    # Attention mechanism melhorado
    try:
        # Usar GlobalAveragePooling1D como alternativa mais simples e estável
        attention_output = tf.keras.layers.GlobalAveragePooling1D()(lstm2)
    except Exception:
        # Fallback para flatten se GlobalAveragePooling1D falhar
        attention_output = Flatten()(lstm2)
        attention_output = Dense(128, activation='relu')(attention_output)
    
    # Dense layers
    dense1 = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(attention_output)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.2)(dense1)
    
    dense2 = Dense(32, activation='relu')(dense1)
    dense2 = BatchNormalization()(dense2)
    
    # Output layer
    if output_steps == 1:
        outputs = Dense(1)(dense2)
    else:
        outputs = Dense(output_steps)(dense2)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Otimizador com learning rate decay
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    model.compile(
        optimizer=optimizer,
        loss='huber',  # Mais robusto a outliers que MSE
        metrics=['mae', 'mse']
    )
    
    return model

# === Versão simplificada do modelo LSTM ===
def criar_modelo_lstm_simplificado(input_shape):
    """Cria um modelo LSTM robusto e simplificado"""
    modelo = Sequential([
        # Primeira camada LSTM
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        BatchNormalization(),
        
        # Segunda camada LSTM
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        
        # Terceira camada LSTM
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),
        
        # Dense layers
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=0.001)
    modelo.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
    
    return modelo

# === Função de Treinamento Aprimorada ===
def treinar_modelo_ticker_avancado(ticker, nome_ticker):
    """Treina modelos avançados para um ticker específico"""
    print(f"\n{'='*80}")
    print(f"🚀 Treinando modelos avançados para {nome_ticker} ({ticker})")
    print(f"{'='*80}")
    
    try:
        # Coleta de dados estendida
        print("📊 Coletando dados históricos...")
        fim = datetime.now()
        inicio = fim - timedelta(days=365*10)  # 10 anos de dados
        
        dados = yf.download(ticker, start=inicio, end=fim, progress=False)
        
        if len(dados) < 500:
            print(f"❌ Dados insuficientes para {ticker} (mínimo: 500 dias)")
            return None
        
        # Preparar dados
        if not isinstance(dados.index, pd.DatetimeIndex):
            dados = dados.reset_index()
            if 'Date' in dados.columns:
                dados.set_index('Date', inplace=True)
        
        print(f"✅ {len(dados)} dias de dados coletados")
        
        # Adicionar indicadores técnicos avançados
        print("📈 Calculando indicadores técnicos avançados...")
        dados = adicionar_indicadores_tecnicos_avancados(dados)
        
        if len(dados) < 300:
            print(f"❌ Dados insuficientes após calcular indicadores")
            return None
        
        # Seleção de features otimizada
        # Vamos usar todas as features disponíveis para o modelo decidir quais são importantes
        features_to_exclude = ['Open', 'High', 'Low', 'Adj Close', 'DayOfWeek', 'DayOfMonth', 'MonthOfYear', 'Quarter']
        features = [col for col in dados.columns if col not in features_to_exclude]
        
        print(f"📊 Total de features: {len(features)}")
        
        dados_features = dados[features].values
        
        # Experimentar com diferentes scalers
        print("🔄 Testando diferentes métodos de normalização...")
        scalers = {
            'minmax': MinMaxScaler(),
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        
        best_results = None
        best_scaler_name = None
        
        for scaler_name, scaler in scalers.items():
            print(f"\n   Testando {scaler_name} scaler...")
            
            # Normalização
            dados_normalizados = scaler.fit_transform(dados_features)
            
            # Criar janelas com horizonte de previsão
            janela = 60  # 60 dias de histórico
            horizonte = 1  # Prever 1 dia à frente
            X, y = criar_janelas_avancadas(dados_normalizados, janela, horizonte)
            
            if len(X) < 200:
                continue
            
            # Dividir dados com validação temporal
            # 60% treino, 20% validação, 20% teste
            tamanho_treino = int(0.6 * len(X))
            tamanho_val = int(0.2 * len(X))
            
            X_train = X[:tamanho_treino]
            y_train = y[:tamanho_treino].reshape(-1, horizonte)
            X_val = X[tamanho_treino:tamanho_treino+tamanho_val]
            y_val = y[tamanho_treino:tamanho_treino+tamanho_val].reshape(-1, horizonte)
            X_test = X[tamanho_treino+tamanho_val:]
            y_test = y[tamanho_treino+tamanho_val:].reshape(-1, horizonte)
            
            # Data Augmentation no conjunto de treino
            print("   📊 Aplicando data augmentation...")
            X_train_aug, y_train_aug = augment_time_series(X_train, y_train, noise_factor=0.005)
            
            # Criar modelo
            try:
                modelo = criar_modelo_attention_lstm((X_train.shape[1], X_train.shape[2]), horizonte)
            except Exception as e:
                print(f"   ⚠️ Usando modelo simplificado devido a: {str(e)}")
                modelo = criar_modelo_lstm_simplificado((X_train.shape[1], X_train.shape[2]))
            
            # Callbacks avançados
            checkpoint_path = f'checkpoints/{ticker}_{scaler_name}_best.keras'
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, mode='min'),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001, mode='min'),
                ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')
            ]
            
            # Treinamento
            history = modelo.fit(
                X_train_aug, y_train_aug,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            # Avaliação
            y_pred = modelo.predict(X_test, verbose=0)
            
            # Desnormalizar apenas a coluna Close (índice 0)
            template_test = np.zeros((len(y_test), len(features)))
            template_pred = np.zeros((len(y_pred), len(features)))
            
            if horizonte == 1:
                template_test[:, 0] = y_test.ravel()
                template_pred[:, 0] = y_pred.ravel()
            else:
                template_test[:, 0] = y_test[:, 0]
                template_pred[:, 0] = y_pred[:, 0]
            
            y_test_real = scaler.inverse_transform(template_test)[:, 0]
            y_pred_real = scaler.inverse_transform(template_pred)[:, 0]
            
            # Métricas
            rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
            mae = mean_absolute_error(y_test_real, y_pred_real)
            r2 = r2_score(y_test_real, y_pred_real)
            mape = np.mean(np.abs((y_test_real - y_pred_real) / y_test_real)) * 100
            
            # Calcular direção correta (para trading)
            if len(y_test_real) > 1:
                direction_accuracy = np.mean(
                    (y_test_real[1:] > y_test_real[:-1]) == 
                    (y_pred_real[1:] > y_pred_real[:-1])
                ) * 100
            else:
                direction_accuracy = 0
            
            current_results = {
                'scaler': scaler_name,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'direction_accuracy': direction_accuracy,
                'model': modelo,
                'scaler_obj': scaler,
                'history': history,
                'y_test_real': y_test_real,
                'y_pred_real': y_pred_real
            }
            
            print(f"   📈 Resultados {scaler_name}:")
            print(f"      RMSE: R$ {rmse:.2f}")
            print(f"      MAE: R$ {mae:.2f}")
            print(f"      R²: {r2:.4f}")
            print(f"      MAPE: {mape:.2f}%")
            print(f"      Acurácia de direção: {direction_accuracy:.2f}%")
            
            if best_results is None or r2 > best_results['r2']:
                best_results = current_results
                best_scaler_name = scaler_name
        
        if best_results is None:
            print(f"❌ Nenhum modelo válido foi treinado para {ticker}")
            return None
        
        print(f"\n✅ Melhor scaler: {best_scaler_name}")
        
        # Salvar melhor modelo
        print("💾 Salvando melhor modelo...")
        best_results['model'].save(f'models/{ticker}_advanced_model.keras')
        joblib.dump(best_results['scaler_obj'], f'scalers/{ticker}_advanced_scaler.pkl')
        
        # Salvar métricas detalhadas
        metricas = {
            'ticker': ticker,
            'nome': nome_ticker,
            'rmse': best_results['rmse'],
            'mae': best_results['mae'],
            'r2': best_results['r2'],
            'mape': best_results['mape'],
            'direction_accuracy': best_results['direction_accuracy'],
            'scaler_type': best_scaler_name,
            'data_treino': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'janela': janela,
            'horizonte': horizonte,
            'num_features': len(features),
            'features': features
        }
        joblib.dump(metricas, f'metrics/{ticker}_advanced_metrics.pkl')
        
        # Visualização aprimorada
        plt.figure(figsize=(20, 12))
        
        # 1. Curvas de aprendizado
        plt.subplot(3, 2, 1)
        plt.plot(best_results['history'].history['loss'], label='Treino', linewidth=2)
        plt.plot(best_results['history'].history['val_loss'], label='Validação', linewidth=2)
        plt.title(f'Curvas de Perda - {nome_ticker}', fontsize=12, fontweight='bold')
        plt.xlabel('Épocas')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. MAE durante treinamento
        plt.subplot(3, 2, 2)
        plt.plot(best_results['history'].history['mae'], label='Treino MAE', linewidth=2)
        plt.plot(best_results['history'].history['val_mae'], label='Validação MAE', linewidth=2)
        plt.title(f'MAE durante Treinamento - {nome_ticker}', fontsize=12, fontweight='bold')
        plt.xlabel('Épocas')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Previsão vs Real (últimos 100 pontos)
        plt.subplot(3, 2, 3)
        n_points = min(100, len(best_results['y_test_real']))
        indices = range(n_points)
        plt.plot(indices, best_results['y_test_real'][-n_points:], 'o-', label='Real', linewidth=2, markersize=4)
        plt.plot(indices, best_results['y_pred_real'][-n_points:], 's-', label='Previsto', linewidth=2, markersize=3, alpha=0.8)
        plt.title(f'Previsão vs Real (Últimos {n_points} dias)', fontsize=12, fontweight='bold')
        plt.xlabel('Dias')
        plt.ylabel('Preço (R$)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Erro de previsão
        plt.subplot(3, 2, 4)
        erros = best_results['y_test_real'] - best_results['y_pred_real']
        plt.hist(erros, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.title(f'Distribuição dos Erros', fontsize=12, fontweight='bold')
        plt.xlabel('Erro (R$)')
        plt.ylabel('Frequência')
        plt.grid(True, alpha=0.3)
        
        # 5. Scatter plot
        plt.subplot(3, 2, 5)
        plt.scatter(best_results['y_test_real'], best_results['y_pred_real'], alpha=0.5, s=20)
        plt.plot([best_results['y_test_real'].min(), best_results['y_test_real'].max()], 
                 [best_results['y_test_real'].min(), best_results['y_test_real'].max()], 
                 'r--', linewidth=2, label='Perfeito')
        plt.title(f'Real vs Previsto', fontsize=12, fontweight='bold')
        plt.xlabel('Preço Real (R$)')
        plt.ylabel('Preço Previsto (R$)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Métricas resumidas
        plt.subplot(3, 2, 6)
        plt.text(0.1, 0.9, f'Métricas de Desempenho - {nome_ticker}', fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
        plt.text(0.1, 0.7, f'RMSE: R$ {best_results["rmse"]:.2f}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.6, f'MAE: R$ {best_results["mae"]:.2f}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.5, f'R²: {best_results["r2"]:.4f}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.4, f'MAPE: {best_results["mape"]:.2f}%', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.3, f'Acurácia Direção: {best_results["direction_accuracy"]:.2f}%', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.2, f'Melhor Scaler: {best_scaler_name}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.1, f'Features: {len(features)}', fontsize=12, transform=plt.gca().transAxes)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'models/{ticker}_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ Modelo avançado para {nome_ticker} treinado com sucesso!")
        print(f"   📊 R² Score: {best_results['r2']:.4f}")
        print(f"   📈 Acurácia de direção: {best_results['direction_accuracy']:.2f}%")
        
        return metricas
        
    except Exception as e:
        print(f"❌ Erro ao treinar {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# === Função para Ensemble Prediction ===
def fazer_previsao_ensemble(ticker, dias_futuros=5):
    """Faz previsões usando ensemble de modelos"""
    try:
        # Carregar dados recentes
        fim = datetime.now()
        inicio = fim - timedelta(days=365)
        dados = yf.download(ticker, start=inicio, end=fim, progress=False)
        
        # Preparar dados
        dados = adicionar_indicadores_tecnicos_avancados(dados)
        
        # Carregar scaler e modelo
        scaler = joblib.load(f'scalers/{ticker}_advanced_scaler.pkl')
        modelo = tf.keras.models.load_model(f'models/{ticker}_advanced_model.keras')
        
        # Preparar features
        metricas = joblib.load(f'metrics/{ticker}_advanced_metrics.pkl')
        features = metricas['features']
        
        dados_features = dados[features].values
        dados_normalizados = scaler.transform(dados_features)
        
        # Criar janela para previsão
        janela = metricas['janela']
        ultima_janela = dados_normalizados[-janela:].reshape(1, janela, -1)
        
        # Fazer previsões iterativas
        previsoes = []
        janela_atual = ultima_janela.copy()
        
        for _ in range(dias_futuros):
            # Prever próximo valor
            pred_normalizado = modelo.predict(janela_atual, verbose=0)
            
            # Desnormalizar
            template = np.zeros((1, len(features)))
            template[0, 0] = pred_normalizado[0, 0]
            pred_real = scaler.inverse_transform(template)[0, 0]
            previsoes.append(pred_real)
            
            # Atualizar janela (sliding window)
            nova_linha = np.zeros((1, len(features)))
            nova_linha[0, 0] = pred_normalizado[0, 0]
            janela_atual = np.concatenate([janela_atual[:, 1:, :], nova_linha.reshape(1, 1, -1)], axis=1)
        
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

def deve_retreinar_modelo(ticker, threshold_r2=0.7, threshold_days=30):
    """Decide se um modelo deve ser retreinado baseado na performance e idade"""
    modelos_existentes = verificar_modelos_existentes()
    
    if ticker not in modelos_existentes:
        return True, "Modelo não existe"
    
    metrics = modelos_existentes[ticker]
    
    # Verificar performance
    if metrics['r2'] < threshold_r2:
        return True, f"R² baixo: {metrics['r2']:.4f} < {threshold_r2}"
    
    # Verificar idade do modelo
    try:
        data_treino = datetime.strptime(metrics['data_treino'], '%Y-%m-%d %H:%M:%S')
        dias_desde_treino = (datetime.now() - data_treino).days
        if dias_desde_treino > threshold_days:
            return True, f"Modelo antigo: {dias_desde_treino} dias"
    except Exception:
        return True, "Data de treino inválida"
    
    return False, f"Modelo atual OK (R²: {metrics['r2']:.4f})"

# === Executar Treinamento Avançado ===
print("🚀 INICIANDO TREINAMENTO AVANÇADO DOS MODELOS")
print(f"📅 Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📊 Total de tickers: {len(TICKERS)}")
print("\n⚡ Este processo usa técnicas avançadas e pode demorar mais tempo...")

# Verificar modelos existentes
print("\n🔍 Verificando modelos existentes...")
modelos_existentes = verificar_modelos_existentes()

if modelos_existentes:
    print(f"✅ Encontrados {len(modelos_existentes)} modelos existentes:")
    for ticker, metrics in modelos_existentes.items():
        print(f"   {ticker}: R² = {metrics['r2']:.4f}, Treino: {metrics['data_treino']}")

# Determinar quais modelos treinar
modelos_para_treinar = []
modelos_pulados = []

for ticker, nome in TICKERS.items():
    deve_treinar, motivo = deve_retreinar_modelo(ticker)
    if deve_treinar:
        modelos_para_treinar.append((ticker, nome, motivo))
    else:
        modelos_pulados.append((ticker, nome, motivo))

if modelos_pulados:
    print(f"\n⏭️ Pulando {len(modelos_pulados)} modelos que já estão bons:")
    for ticker, nome, motivo in modelos_pulados:
        print(f"   {nome} ({ticker}): {motivo}")

if not modelos_para_treinar:
    print("\n✅ Todos os modelos estão atualizados e com boa performance!")
    print("💡 Use force_retrain=True se quiser retreinar todos os modelos")
else:
    print(f"\n🔄 Treinando {len(modelos_para_treinar)} modelo(s):")
    for ticker, nome, motivo in modelos_para_treinar:
        print(f"   {nome} ({ticker}): {motivo}")

resultados = []
total_modelos = len(modelos_para_treinar)

for i, (ticker, nome, motivo) in enumerate(modelos_para_treinar, 1):
    print(f"\n📊 Progresso: {i}/{total_modelos} modelos")
    print(f"🎯 Motivo do retreino: {motivo}")
    
    try:
        resultado = treinar_modelo_ticker_avancado(ticker, nome)
        if resultado:
            resultados.append(resultado)
            logger.info(f"✅ Modelo {ticker} treinado com sucesso")
        else:
            logger.warning(f"⚠️ Falha ao treinar modelo {ticker}")
        
        # Limpeza de memória entre modelos
        gc.collect()
        tf.keras.backend.clear_session()
        
        # Pequena pausa entre tickers para não sobrecarregar a API
        import time
        time.sleep(2)
        
    except Exception as e:
        logger.error(f"❌ Erro crítico ao treinar {ticker}: {str(e)}")
        continue

# Adicionar modelos existentes que não foram retreinados aos resultados
for ticker, nome in modelos_pulados:
    try:
        metrics_path = f'metrics/{ticker}_advanced_metrics.pkl'
        metrics = joblib.load(metrics_path)
        resultados.append(metrics)
    except Exception as e:
        logger.warning(f"⚠️ Erro ao carregar métricas de {ticker}: {e}")

# === Resumo Final Detalhado ===
print("\n" + "="*80)
print("📊 RESUMO DO TREINAMENTO AVANÇADO")
print("="*80)

if resultados:
    print(f"\n✅ Modelos treinados com sucesso: {len(resultados)}/{len(TICKERS)}")
    
    # Top modelos por R²
    print("\n🏆 Top 5 modelos por R²:")
    resultados_r2 = sorted(resultados, key=lambda x: x['r2'], reverse=True)
    for i, res in enumerate(resultados_r2[:5]):
        print(f"{i+1}. {res['nome']} ({res['ticker']}): R² = {res['r2']:.4f}")
    
    # Top modelos por acurácia de direção
    print("\n🎯 Top 5 modelos por Acurácia de Direção:")
    resultados_dir = sorted(resultados, key=lambda x: x['direction_accuracy'], reverse=True)
    for i, res in enumerate(resultados_dir[:5]):
        print(f"{i+1}. {res['nome']} ({res['ticker']}): {res['direction_accuracy']:.2f}%")
    
    # Estatísticas gerais
    avg_r2 = np.mean([r['r2'] for r in resultados])
    avg_mape = np.mean([r['mape'] for r in resultados])
    avg_direction = np.mean([r['direction_accuracy'] for r in resultados])
    
    print(f"\n📊 Estatísticas Gerais:")
    print(f"   R² médio: {avg_r2:.4f}")
    print(f"   MAPE médio: {avg_mape:.2f}%")
    print(f"   Acurácia de direção média: {avg_direction:.2f}%")
    
    # Recomendações
    print("\n💡 Recomendações para uso:")
    print("   1. Modelos com R² > 0.7 são considerados bons")
    print("   2. Acurácia de direção > 55% é útil para trading")
    print("   3. MAPE < 5% indica previsões precisas")
    print("   4. Use ensemble de modelos para melhores resultados")
    print("   5. Reavalie modelos mensalmente com novos dados")
    
else:
    print("\n❌ Nenhum modelo foi treinado. Verifique os logs.")

print("\n✅ Processo concluído!")
print("📁 Modelos salvos em: ./models/")
print("📁 Scalers salvos em: ./scalers/")
print("📁 Métricas salvas em: ./metrics/")
print("📁 Checkpoints salvos em: ./checkpoints/")
print("📁 Gráficos salvos em: ./models/")

# === Exemplo de uso para previsão ===
print("\n" + "="*80)
print("🔮 EXEMPLO DE PREVISÃO")
print("="*80)

if resultados:
    # Pegar o melhor modelo por R²
    melhor_modelo = resultados_r2[0]
    ticker_exemplo = melhor_modelo['ticker']
    nome_exemplo = melhor_modelo['nome']
    
    print(f"\nFazendo previsão para {nome_exemplo} ({ticker_exemplo}) - próximos 5 dias:")
    
    previsoes, ultima_data = fazer_previsao_ensemble(ticker_exemplo, dias_futuros=5)
    
    if previsoes:
        print(f"Última data conhecida: {ultima_data.strftime('%Y-%m-%d')}")
        for i, prev in enumerate(previsoes, 1):
            data_prev = ultima_data + timedelta(days=i)
            print(f"   {data_prev.strftime('%Y-%m-%d')}: R$ {prev:.2f}")
    else:
        print("❌ Erro ao fazer previsão")

print("\n🎉 Script finalizado com sucesso!")
