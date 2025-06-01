import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, GRU, Bidirectional, Concatenate, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import ta
import joblib
import os
from datetime import datetime, timedelta
import warnings
import gc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Configure TensorFlow for better memory usage
try:
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
}

# Índices de mercado e correlações
MARKET_INDICES = {
    '^BVSP': 'Ibovespa',
    '^DJI': 'Dow Jones',
    'CL=F': 'Petróleo WTI',  # Importante para Petrobras
    'BRL=X': 'USD/BRL'
}

# Diretórios
os.makedirs('models', exist_ok=True)
os.makedirs('scalers', exist_ok=True)
os.makedirs('metrics', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# === Função para adicionar indicadores técnicos essenciais ===
def adicionar_indicadores_tecnicos_essenciais(df):
    """Adiciona apenas indicadores técnicos mais relevantes"""
    df = df.copy()
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # Médias móveis essenciais
    df['SMA_5'] = ta.trend.sma_indicator(close, window=5)
    df['SMA_20'] = ta.trend.sma_indicator(close, window=20)
    df['EMA_9'] = ta.trend.ema_indicator(close, window=9)
    df['EMA_21'] = ta.trend.ema_indicator(close, window=21)
    
    # Razões de médias móveis (mais estáveis)
    df['SMA_ratio'] = df['SMA_5'] / df['SMA_20']
    df['Price_to_SMA20'] = close / df['SMA_20']
    
    # RSI
    df['RSI'] = ta.momentum.rsi(close, window=14)
    
    # MACD
    macd = ta.trend.MACD(close)
    df['MACD_diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20)
    df['BB_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / close
    df['BB_position'] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    
    # Volume indicators
    df['Volume_ratio'] = volume / volume.rolling(20).mean()
    df['OBV'] = ta.volume.on_balance_volume(close, volume)
    df['OBV_EMA'] = ta.trend.ema_indicator(df['OBV'], window=20)
    
    # Volatilidade
    df['ATR'] = ta.volatility.average_true_range(high, low, close, window=14)
    df['Volatility'] = close.pct_change().rolling(20).std()
    
    # Returns
    df['Return_1'] = close.pct_change(1)
    df['Return_5'] = close.pct_change(5)
    df['Return_20'] = close.pct_change(20)
    
    # Price patterns
    df['HL_ratio'] = (high - low) / close
    df['CO_ratio'] = (close - df['Open']) / df['Open']
    
    # Trend indicators
    df['Trend_20'] = (close - close.shift(20)) / close.shift(20)
    df['Above_SMA20'] = (close > df['SMA_20']).astype(int)
    
    return df

# === Função para coletar dados de mercado ===
def coletar_dados_mercado(inicio, fim):
    """Coleta dados dos índices de mercado para usar como features adicionais"""
    market_data = {}
    valid_data = []
    
    for symbol, name in MARKET_INDICES.items():
        try:
            data = yf.download(symbol, start=inicio, end=fim, progress=False)
            if len(data) > 0 and 'Close' in data.columns:
                # Criar DataFrame temporário com os dados deste símbolo
                temp_df = pd.DataFrame(index=data.index)
                temp_df[f'{name}_Close'] = data['Close']
                temp_df[f'{name}_Return'] = data['Close'].pct_change()
                
                valid_data.append(temp_df)
                logger.info(f"✅ Dados de {name} coletados")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao coletar {name}: {e}")
    
    # Se temos dados válidos, concatenar todos
    if valid_data:
        result_df = valid_data[0]
        for df in valid_data[1:]:
            result_df = result_df.join(df, how='outer')
        return result_df
    else:
        # Retornar DataFrame vazio se não conseguimos coletar nenhum dado
        logger.warning("⚠️ Nenhum dado de mercado foi coletado")
        return pd.DataFrame()

# === Seleção inteligente de features ===
def selecionar_features_importantes(X, y, n_features=30):
    """Seleciona as features mais importantes usando mutual information"""
    # Remover NaN
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_clean = X[mask]
    y_clean = y[mask]
    
    if len(X_clean) < 100:
        return list(range(min(n_features, X.shape[1])))
    
    selector = SelectKBest(score_func=mutual_info_regression, k=min(n_features, X_clean.shape[1]))
    selector.fit(X_clean, y_clean)
    
    return selector.get_support(indices=True).tolist()

# === Criar conjunto de validação walk-forward ===
def criar_validacao_walk_forward(X, y, n_splits=5, test_size=60):
    """Cria múltiplos conjuntos de validação temporal"""
    splits = []
    total_size = len(X)
    
    for i in range(n_splits):
        test_end = total_size - (i * test_size)
        test_start = test_end - test_size
        val_end = test_start
        val_start = val_end - test_size
        train_end = val_start
        
        if train_end < 200:  # Mínimo de dados para treino
            break
            
        train_idx = list(range(0, train_end))
        val_idx = list(range(val_start, val_end))
        test_idx = list(range(test_start, test_end))
        
        splits.append((train_idx, val_idx, test_idx))
    
    return splits

# === Modelos mais simples e robustos ===
def criar_modelo_lstm_robusto(input_shape, learning_rate=0.001):
    """Cria um modelo LSTM mais simples e robusto"""
    model = Sequential([
        # Primeira camada LSTM
        LSTM(50, return_sequences=True, input_shape=input_shape,
             kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        Dropout(0.3),
        
        # Segunda camada LSTM
        LSTM(30, return_sequences=False,
             kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        Dropout(0.3),
        
        # Camadas densas
        Dense(20, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='relu'),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

def criar_modelo_gru(input_shape, learning_rate=0.001):
    """Cria um modelo GRU como alternativa"""
    model = Sequential([
        GRU(40, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        GRU(20, return_sequences=False),
        Dropout(0.2),
        Dense(15, activation='relu'),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

def criar_modelo_cnn_lstm(input_shape, learning_rate=0.001):
    """Cria um modelo híbrido CNN-LSTM"""
    from tensorflow.keras.layers import Conv1D, MaxPooling1D
    
    model = Sequential([
        # Camadas convolucionais para extrair features locais
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        
        # LSTM para capturar dependências temporais
        LSTM(30, return_sequences=False),
        Dropout(0.3),
        
        # Camadas densas
        Dense(20, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

# === Função principal de treinamento melhorada ===
def treinar_modelo_ticker_melhorado(ticker, nome_ticker):
    """Treina modelos melhorados para um ticker específico"""
    print(f"\n{'='*80}")
    print(f"🚀 Treinando modelos melhorados para {nome_ticker} ({ticker})")
    print(f"{'='*80}")
    
    try:
        # Coleta de dados
        print("📊 Coletando dados históricos...")
        fim = datetime.now()
        inicio = fim - timedelta(days=365*7)  # 7 anos de dados
        
        # Dados do ticker
        dados = yf.download(ticker, start=inicio, end=fim, progress=False)
        
        # Flatten multi-level columns if they exist
        if isinstance(dados.columns, pd.MultiIndex):
            dados.columns = [col[0] if isinstance(col, tuple) else col for col in dados.columns]
        
        if len(dados) < 500:
            print(f"❌ Dados insuficientes para {ticker}")
            return None
        
        # Dados de mercado
        print("📈 Coletando dados de mercado correlacionados...")
        dados_mercado = coletar_dados_mercado(inicio, fim)
        
        # Alinhar índices e juntar dados
        if len(dados_mercado) > 0:
            dados = dados.join(dados_mercado, how='left')
        
        # Adicionar indicadores técnicos
        print("📊 Calculando indicadores técnicos...")
        dados = adicionar_indicadores_tecnicos_essenciais(dados)
        
        # Preencher valores faltantes
        dados.fillna(method='ffill', inplace=True)
        dados.dropna(inplace=True)
        
        if len(dados) < 300:
            print(f"❌ Dados insuficientes após processamento")
            return None
        
        # Preparar features e target
        # IMPORTANTE: Separar o target (Close) das features
        target_col = 'Close'
        feature_cols = [col for col in dados.columns if col not in ['Close', 'Open', 'High', 'Low', 'Adj Close']]
        
        print(f"📊 Features selecionadas: {len(feature_cols)}")
        
        # Normalização separada para features e target
        scaler_features = RobustScaler()
        scaler_target = MinMaxScaler()
        
        features_scaled = scaler_features.fit_transform(dados[feature_cols])
        target_scaled = scaler_target.fit_transform(dados[[target_col]])
        
        # Seleção de features importantes
        print("🎯 Selecionando features mais importantes...")
        important_indices = selecionar_features_importantes(
            features_scaled, 
            target_scaled.ravel(), 
            n_features=25
        )
        
        features_selected = features_scaled[:, important_indices]
        selected_feature_names = [feature_cols[i] for i in important_indices]
        print(f"✅ Features selecionadas: {len(selected_feature_names)}")
        
        # Criar sequências temporais
        janela = 30  # Reduzir janela para 30 dias
        X, y = [], []
        
        for i in range(len(features_selected) - janela):
            X.append(features_selected[i:i+janela])
            y.append(target_scaled[i+janela, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"📊 Shape dos dados: X={X.shape}, y={y.shape}")
        
        # Walk-forward validation
        splits = criar_validacao_walk_forward(X, y, n_splits=3)
        
        # Treinar múltiplos modelos
        modelos = {
            'LSTM_Robusto': criar_modelo_lstm_robusto,
            'GRU': criar_modelo_gru,
            'CNN_LSTM': criar_modelo_cnn_lstm
        }
        
        resultados_modelos = {}
        
        for nome_modelo, criar_modelo_func in modelos.items():
            print(f"\n🔧 Treinando modelo {nome_modelo}...")
            
            scores = []
            
            for fold, (train_idx, val_idx, test_idx) in enumerate(splits):
                print(f"   Fold {fold+1}/{len(splits)}...")
                
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_val = X[val_idx]
                y_val = y[val_idx]
                X_test = X[test_idx]
                y_test = y[test_idx]
                
                # Criar e treinar modelo
                model = criar_modelo_func((X_train.shape[1], X_train.shape[2]))
                
                # Callbacks
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001)
                ]
                
                # Treinar
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Avaliar
                y_pred = model.predict(X_test, verbose=0).ravel()
                
                # Desnormalizar
                y_test_real = scaler_target.inverse_transform(y_test.reshape(-1, 1)).ravel()
                y_pred_real = scaler_target.inverse_transform(y_pred.reshape(-1, 1)).ravel()
                
                # Métricas
                rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
                mae = mean_absolute_error(y_test_real, y_pred_real)
                r2 = r2_score(y_test_real, y_pred_real)
                
                # Direção
                if len(y_test_real) > 1:
                    actual_direction = np.diff(y_test_real) > 0
                    pred_direction = np.diff(y_pred_real) > 0
                    direction_accuracy = np.mean(actual_direction == pred_direction) * 100
                else:
                    direction_accuracy = 0
                
                scores.append({
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'direction_accuracy': direction_accuracy,
                    'model': model
                })
            
            # Média dos scores
            avg_scores = {
                'rmse': np.mean([s['rmse'] for s in scores]),
                'mae': np.mean([s['mae'] for s in scores]),
                'r2': np.mean([s['r2'] for s in scores]),
                'direction_accuracy': np.mean([s['direction_accuracy'] for s in scores])
            }
            
            resultados_modelos[nome_modelo] = {
                'scores': avg_scores,
                'best_model': scores[np.argmax([s['r2'] for s in scores])]['model']
            }
            
            print(f"   R² médio: {avg_scores['r2']:.4f}")
            print(f"   Acurácia direção: {avg_scores['direction_accuracy']:.2f}%")
        
        # Selecionar melhor modelo
        melhor_modelo_nome = max(resultados_modelos.keys(), 
                                key=lambda k: resultados_modelos[k]['scores']['r2'])
        melhor_resultado = resultados_modelos[melhor_modelo_nome]
        
        print(f"\n✅ Melhor modelo: {melhor_modelo_nome}")
        print(f"   R²: {melhor_resultado['scores']['r2']:.4f}")
        
        # Treinar modelo final com todos os dados
        print("\n🔄 Treinando modelo final com todos os dados...")
        
        # Dividir em treino/teste final
        split_point = int(0.8 * len(X))
        X_train_final = X[:split_point]
        y_train_final = y[:split_point]
        X_test_final = X[split_point:]
        y_test_final = y[split_point:]
        
        # Criar e treinar modelo final
        if melhor_modelo_nome == 'LSTM_Robusto':
            modelo_final = criar_modelo_lstm_robusto((X.shape[1], X.shape[2]))
        elif melhor_modelo_nome == 'GRU':
            modelo_final = criar_modelo_gru((X.shape[1], X.shape[2]))
        else:
            modelo_final = criar_modelo_cnn_lstm((X.shape[1], X.shape[2]))
        
        # Callbacks para modelo final
        checkpoint_path = f'checkpoints/{ticker}_best_model.keras'
        callbacks_final = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001),
            ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
        ]
        
        # Treinar modelo final
        history_final = modelo_final.fit(
            X_train_final, y_train_final,
            validation_split=0.2,
            epochs=150,
            batch_size=32,
            callbacks=callbacks_final,
            verbose=1
        )
        
        # Avaliação final
        y_pred_final = modelo_final.predict(X_test_final, verbose=0).ravel()
        
        # Desnormalizar
        y_test_real = scaler_target.inverse_transform(y_test_final.reshape(-1, 1)).ravel()
        y_pred_real = scaler_target.inverse_transform(y_pred_final.reshape(-1, 1)).ravel()
        
        # Métricas finais
        rmse_final = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
        mae_final = mean_absolute_error(y_test_real, y_pred_real)
        r2_final = r2_score(y_test_real, y_pred_real)
        mape_final = np.mean(np.abs((y_test_real - y_pred_real) / y_test_real)) * 100
        
        # Direção
        if len(y_test_real) > 1:
            actual_direction = np.diff(y_test_real) > 0
            pred_direction = np.diff(y_pred_real) > 0
            direction_accuracy_final = np.mean(actual_direction == pred_direction) * 100
        else:
            direction_accuracy_final = 0
        
        print(f"\n📊 Métricas finais:")
        print(f"   RMSE: R$ {rmse_final:.2f}")
        print(f"   MAE: R$ {mae_final:.2f}")
        print(f"   R²: {r2_final:.4f}")
        print(f"   MAPE: {mape_final:.2f}%")
        print(f"   Acurácia direção: {direction_accuracy_final:.2f}%")
        
        # Salvar modelo e scalers
        print("\n💾 Salvando modelo e configurações...")
        
        # Salvar com a estrutura esperada pela UI
        modelo_final.save(f'models/{ticker}_advanced_model.keras')
        
        # Criar scaler compatível com o formato original
        # Precisamos salvar um scaler que funcione com todas as features originais
        scaler_completo = MinMaxScaler()
        dados_completos = dados[feature_cols + [target_col]].values
        scaler_completo.fit(dados_completos)
        
        joblib.dump(scaler_completo, f'scalers/{ticker}_advanced_scaler.pkl')
        
        # Salvar métricas e configurações
        metricas = {
            'ticker': ticker,
            'nome': nome_ticker,
            'rmse': rmse_final,
            'mae': mae_final,
            'r2': r2_final,
            'mape': mape_final,
            'direction_accuracy': direction_accuracy_final,
            'scaler_type': 'minmax',
            'data_treino': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'janela': janela,
            'horizonte': 1,
            'num_features': len(feature_cols) + 1,  # +1 para incluir Close
            'features': feature_cols + [target_col],  # Ordem importante!
            'selected_features': selected_feature_names,
            'modelo_tipo': melhor_modelo_nome,
            'important_indices': important_indices
        }
        
        joblib.dump(metricas, f'metrics/{ticker}_advanced_metrics.pkl')
        
        # Visualização
        criar_visualizacao_melhorada(
            ticker, nome_ticker, history_final, 
            y_test_real, y_pred_real, metricas
        )
        
        return metricas
        
    except Exception as e:
        print(f"❌ Erro ao treinar {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def criar_visualizacao_melhorada(ticker, nome_ticker, history, y_test, y_pred, metricas):
    """Cria visualizações melhoradas dos resultados"""
    plt.figure(figsize=(20, 12))
    
    # 1. Curvas de aprendizado
    plt.subplot(3, 2, 1)
    plt.plot(history.history['loss'], label='Treino', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validação', linewidth=2)
    plt.title(f'Curvas de Perda - {nome_ticker}', fontsize=12, fontweight='bold')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. MAE durante treinamento
    plt.subplot(3, 2, 2)
    plt.plot(history.history['mae'], label='Treino MAE', linewidth=2)
    plt.plot(history.history['val_mae'], label='Validação MAE', linewidth=2)
    plt.title(f'MAE durante Treinamento - {nome_ticker}', fontsize=12, fontweight='bold')
    plt.xlabel('Épocas')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Previsão vs Real
    plt.subplot(3, 2, 3)
    n_points = min(100, len(y_test))
    indices = range(n_points)
    plt.plot(indices, y_test[-n_points:], 'o-', label='Real', linewidth=2, markersize=4)
    plt.plot(indices, y_pred[-n_points:], 's-', label='Previsto', linewidth=2, markersize=3, alpha=0.8)
    plt.title(f'Previsão vs Real (Últimos {n_points} dias)', fontsize=12, fontweight='bold')
    plt.xlabel('Dias')
    plt.ylabel('Preço (R$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Distribuição dos erros
    plt.subplot(3, 2, 4)
    erros = y_test - y_pred
    plt.hist(erros, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.axvline(x=np.mean(erros), color='green', linestyle='--', linewidth=2, label=f'Média: {np.mean(erros):.2f}')
    plt.title(f'Distribuição dos Erros', fontsize=12, fontweight='bold')
    plt.xlabel('Erro (R$)')
    plt.ylabel('Frequência')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Scatter plot com linha de tendência
    plt.subplot(3, 2, 5)
    plt.scatter(y_test, y_pred, alpha=0.5, s=20)
    
    # Linha de regressão
    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_test, p(y_test), "r--", alpha=0.8, label=f'Tendência: y={z[0]:.2f}x+{z[1]:.2f}')
    
    # Linha perfeita
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'g--', linewidth=2, label='Perfeito')
    
    plt.title(f'Real vs Previsto', fontsize=12, fontweight='bold')
    plt.xlabel('Preço Real (R$)')
    plt.ylabel('Preço Previsto (R$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Métricas e informações
    plt.subplot(3, 2, 6)
    plt.text(0.1, 0.9, f'Métricas de Desempenho - {nome_ticker}', fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
    plt.text(0.1, 0.75, f'Modelo: {metricas.get("modelo_tipo", "LSTM")}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.65, f'RMSE: R$ {metricas["rmse"]:.2f}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.55, f'MAE: R$ {metricas["mae"]:.2f}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.45, f'R²: {metricas["r2"]:.4f}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.35, f'MAPE: {metricas["mape"]:.2f}%', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.25, f'Acurácia Direção: {metricas["direction_accuracy"]:.2f}%', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.15, f'Features selecionadas: {len(metricas.get("selected_features", []))}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.05, f'Janela temporal: {metricas["janela"]} dias', fontsize=12, transform=plt.gca().transAxes)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'models/{ticker}_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# === Função para fazer previsões (compatível com UI existente) ===
def fazer_previsao_ensemble(ticker, dias_futuros=5):
    """Faz previsões usando o modelo treinado"""
    try:
        # Carregar dados recentes
        fim = datetime.now()
        inicio = fim - timedelta(days=365)
        dados = yf.download(ticker, start=inicio, end=fim, progress=False)
        
        # Coletar dados de mercado
        dados_mercado = coletar_dados_mercado(inicio, fim)
        if len(dados_mercado) > 0:
            dados = dados.join(dados_mercado, how='left')
        
        # Adicionar indicadores
        dados = adicionar_indicadores_tecnicos_essenciais(dados)
        dados.fillna(method='ffill', inplace=True)
        dados.dropna(inplace=True)
        
        # Carregar configurações
        metricas = joblib.load(f'metrics/{ticker}_advanced_metrics.pkl')
        scaler = joblib.load(f'scalers/{ticker}_advanced_scaler.pkl')
        modelo = tf.keras.models.load_model(f'models/{ticker}_advanced_model.keras')
        
        # Preparar dados com as features corretas
        features = metricas['features']
        
        # Garantir que temos todas as features necessárias
        features_disponiveis = [f for f in features if f in dados.columns]
        if len(features_disponiveis) < len(features):
            # Criar features faltantes com valores padrão
            for f in features:
                if f not in dados.columns:
                    dados[f] = 0
        
        dados_features = dados[features].values
        dados_normalizados = scaler.transform(dados_features)
        
        # Criar janela
        janela = metricas['janela']
        
        # Se temos features selecionadas, usar apenas elas
        if 'important_indices' in metricas:
            indices = metricas['important_indices']
            # Ajustar índices para excluir a coluna Close das features
            features_indices = [i for i in indices if i < len(features)-1]
            dados_para_modelo = dados_normalizados[:, features_indices]
        else:
            # Usar todas exceto Close
            dados_para_modelo = dados_normalizados[:, :-1]
        
        ultima_janela = dados_para_modelo[-janela:].reshape(1, janela, -1)
        
        # Fazer previsões
        previsoes = []
        for _ in range(dias_futuros):
            pred_normalizado = modelo.predict(ultima_janela, verbose=0)[0, 0]
            
            # Desnormalizar (assumindo que Close é a última coluna)
            pred_array = np.zeros((1, len(features)))
            pred_array[0, -1] = pred_normalizado  # Close é a última
            pred_real = scaler.inverse_transform(pred_array)[0, -1]
            
            previsoes.append(pred_real)
            
            # Atualizar janela (simplificado - usar última linha)
            # Em produção, você precisaria recalcular os indicadores
            nova_linha = dados_para_modelo[-1:].copy()
            ultima_janela = np.concatenate([ultima_janela[:, 1:, :], nova_linha.reshape(1, 1, -1)], axis=1)
        
        return previsoes, dados.index[-1]
        
    except Exception as e:
        print(f"Erro na previsão: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

# === Função principal ===
def main():
    print("🚀 INICIANDO TREINAMENTO MELHORADO DOS MODELOS")
    print(f"📅 Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Total de tickers: {len(TICKERS)}")
    
    resultados = []
    
    for ticker, nome in TICKERS.items():
        print(f"\n{'='*80}")
        resultado = treinar_modelo_ticker_melhorado(ticker, nome)
        if resultado:
            resultados.append(resultado)
            gc.collect()
            tf.keras.backend.clear_session()
    
    # Resumo final
    print("\n" + "="*80)
    print("📊 RESUMO DO TREINAMENTO")
    print("="*80)
    
    if resultados:
        print(f"\n✅ Modelos treinados com sucesso: {len(resultados)}/{len(TICKERS)}")
        
        for res in resultados:
            print(f"\n{res['nome']} ({res['ticker']}):")
            print(f"   Modelo: {res.get('modelo_tipo', 'LSTM')}")
            print(f"   R²: {res['r2']:.4f}")
            print(f"   RMSE: R$ {res['rmse']:.2f}")
            print(f"   Acurácia direção: {res['direction_accuracy']:.2f}%")
        
        # Fazer previsão de exemplo
        if resultados:
            ticker_exemplo = resultados[0]['ticker']
            print(f"\n🔮 Exemplo de previsão para {resultados[0]['nome']} - próximos 5 dias:")
            
            previsoes, ultima_data = fazer_previsao_ensemble(ticker_exemplo, dias_futuros=5)
            
            if previsoes:
                print(f"Última data: {ultima_data.strftime('%Y-%m-%d')}")
                for i, prev in enumerate(previsoes, 1):
                    data_prev = ultima_data + timedelta(days=i)
                    print(f"   {data_prev.strftime('%Y-%m-%d')}: R$ {prev:.2f}")
    
    print("\n✅ Processo finalizado!")

# Executar se for o script principal
if __name__ == "__main__":
    main()