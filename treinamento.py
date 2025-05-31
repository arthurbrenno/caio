
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import ta  # Technical Analysis library
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# === 3. Configuração de Tickers ===
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

# === 4. Diretórios para Salvar Modelos ===
os.makedirs('models', exist_ok=True)
os.makedirs('scalers', exist_ok=True)
os.makedirs('metrics', exist_ok=True)

# === 5. Função para Adicionar Indicadores Técnicos ===
def adicionar_indicadores_tecnicos(df):
    """Adiciona indicadores técnicos ao DataFrame"""
    df = df.copy()

    # Garante que as colunas primárias sejam Series, pegando a primeira coluna se a seleção resultar em DataFrame.
    # Isso torna a função mais robusta, especialmente se yf.download retornar múltiplas colunas
    # para um nome padrão (ex: 'Volume' devido a alguma peculiaridade ou duplicata).

    _close_col = df['Close']
    if isinstance(_close_col, pd.DataFrame):
        close_prices = _close_col.iloc[:, 0].squeeze()
    else:
        close_prices = _close_col.squeeze()

    _volume_col = df['Volume']
    if isinstance(_volume_col, pd.DataFrame):
        # Parte crucial: se df['Volume'] selecionar múltiplas colunas,
        # pegamos a primeira, assumindo que é o comportamento implícito da TA-Lib ou o esperado.
        volume_series_for_ta = _volume_col.iloc[:, 0].squeeze()
    else:
        volume_series_for_ta = _volume_col.squeeze()

    _high_col = df['High']
    if isinstance(_high_col, pd.DataFrame):
        high_prices = _high_col.iloc[:, 0].squeeze()
    else:
        high_prices = _high_col.squeeze()

    _low_col = df['Low']
    if isinstance(_low_col, pd.DataFrame):
        low_prices = _low_col.iloc[:, 0].squeeze()
    else:
        low_prices = _low_col.squeeze()

    # Médias Móveis
    df['SMA_7'] = ta.trend.sma_indicator(close_prices, window=7)
    df['SMA_21'] = ta.trend.sma_indicator(close_prices, window=21)
    df['EMA_9'] = ta.trend.ema_indicator(close_prices, window=9)

    # RSI
    df['RSI'] = ta.momentum.rsi(close_prices, window=14)

    # MACD
    macd_obj = ta.trend.MACD(close_prices)
    df['MACD'] = macd_obj.macd()
    df['MACD_signal'] = macd_obj.macd_signal()

    # Bollinger Bands
    bollinger_obj = ta.volatility.BollingerBands(close_prices)
    df['BB_upper'] = bollinger_obj.bollinger_hband()
    df['BB_lower'] = bollinger_obj.bollinger_lband()
    df['BB_width'] = df['BB_upper'] - df['BB_lower']

    # Volume indicators
    # Usa 'volume_series_for_ta' que agora é garantidamente uma única Series
    df['Volume_SMA'] = ta.trend.sma_indicator(volume_series_for_ta, window=10)

    # Garante que df['Volume_SMA'] também seja tratado como Series para a divisão
    # A TA-Lib deve retornar Series, mas .squeeze() é uma salvaguarda.
    volume_sma_as_series = df['Volume_SMA'].squeeze()

    # Calcula Volume_ratio usando a mesma 'volume_series_for_ta'
    # Esta é a correção principal para o erro original
    df['Volume_ratio'] = volume_series_for_ta / volume_sma_as_series

    # Price features
    df['High_Low_pct'] = (high_prices - low_prices) / close_prices * 100
    df['Price_change'] = close_prices.pct_change()

    # Remove NaN values
    df.dropna(inplace=True)

    return df

# === 6. Função de Criação de Janelas Temporais ===
def criar_janelas(dados, janela=10):
    """Cria janelas temporais para treino"""
    X, y = [], []
    for i in range(len(dados) - janela):
        X.append(dados[i:i+janela])
        y.append(dados[i+janela][0])  # Prevê o Close normalizado
    return np.array(X), np.array(y)

# === 7. Função para Criar Modelo LSTM Melhorado ===
def criar_modelo_lstm(input_shape):
    """Cria um modelo LSTM otimizado"""
    modelo = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        BatchNormalization(),

        LSTM(64, return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),

        LSTM(32, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),

        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=0.001)
    modelo.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return modelo

# === 8. Função Principal de Treinamento ===
def treinar_modelo_ticker(ticker, nome_ticker):
    """Treina um modelo para um ticker específico"""
    print(f"\n{'='*60}")
    print(f"Treinando modelo para {nome_ticker} ({ticker})")
    print(f"{'='*60}")

    try:
        # Coleta de dados
        print("📊 Coletando dados...")
        fim = datetime.now()
        inicio = fim - timedelta(days=365*5)  # 5 anos de dados (reduzido para melhor performance)

        # Download com progress=False para evitar problemas
        dados = yf.download(ticker, start=inicio, end=fim, progress=False)

        # Verificar se temos dados suficientes
        if len(dados) < 100:
            print(f"❌ Dados insuficientes para {ticker}")
            return None

        # Resetar index para garantir que seja um DataFrame limpo
        # yf.download já retorna com Date como index se não houver group_by.
        # Se Date não for index, as linhas abaixo o configuram.
        if not isinstance(dados.index, pd.DatetimeIndex):
            dados = dados.reset_index()
            if 'Date' in dados.columns:
                dados.set_index('Date', inplace=True)
            elif 'Datetime' in dados.columns: # Algumas versões/tickers podem usar Datetime
                 dados.set_index('Datetime', inplace=True)
            else:
                print(f"❌ Coluna de data não encontrada para {ticker}")
                return None

        # Adicionar indicadores técnicos
        print("📈 Calculando indicadores técnicos...")
        dados = adicionar_indicadores_tecnicos(dados)

        # Verificar se ainda temos dados após remover NaN
        if len(dados) < 100:
            print(f"❌ Dados insuficientes após calcular indicadores para {ticker}")
            return None

        # Selecionar features
        features = ['Close', 'Volume', 'SMA_7', 'SMA_21', 'EMA_9', 'RSI',
                    'MACD', 'BB_width', 'Volume_ratio', 'High_Low_pct']
        dados_features = dados[features].values

        # Normalização
        print("🔄 Normalizando dados...")
        scaler = MinMaxScaler()
        dados_normalizados = scaler.fit_transform(dados_features)

        # Criar janelas
        janela = 20
        X, y = criar_janelas(dados_normalizados, janela)

        # Verificar se temos dados suficientes
        if len(X) < 100: # Checagem após criação de janelas
            print(f"❌ Dados insuficientes após criar janelas para {ticker} (X_len: {len(X)})")
            return None

        # Divisão dos dados
        tamanho_treino = int(0.7 * len(X))
        tamanho_val = int(0.15 * len(X))

        X_train = X[:tamanho_treino]
        y_train = y[:tamanho_treino]
        X_val = X[tamanho_treino:tamanho_treino+tamanho_val]
        y_val = y[tamanho_treino:tamanho_treino+tamanho_val]
        X_test = X[tamanho_treino+tamanho_val:]
        y_test = y[tamanho_treino+tamanho_val:]

        # Verificar se os splits não estão vazios
        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            print(f"❌ Dados insuficientes para divisão treino/val/teste para {ticker}")
            return None

        # Criar e treinar modelo
        print("🧠 Criando e treinando modelo LSTM...")
        modelo = criar_modelo_lstm((X_train.shape[1], X_train.shape[2]))

        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

        # Treinamento
        history = modelo.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,  # Reduzido para treinar mais rápido
            batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=0 # Alterado para 0 para reduzir output durante loop de treino
        )

        # Avaliação
        print("📊 Avaliando modelo...")
        y_pred = modelo.predict(X_test, verbose=0)

        # Desnormalizar para métricas reais
        # Criar um array template para inverse_transform, garantindo que y_test e y_pred tenham a mesma forma que 'Close' original
        # A coluna 'Close' é a primeira (índice 0) nas 'features'

        # Para y_test_real
        template_test = np.zeros((len(y_test), len(features)))
        template_test[:, 0] = y_test.ravel() # y_test é o 'Close' normalizado
        y_test_real = scaler.inverse_transform(template_test)[:, 0]

        # Para y_pred_real
        template_pred = np.zeros((len(y_pred), len(features)))
        template_pred[:, 0] = y_pred.ravel() # y_pred é o 'Close' normalizado previsto
        y_pred_real = scaler.inverse_transform(template_pred)[:, 0]

        rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
        mae = mean_absolute_error(y_test_real, y_pred_real)
        r2 = r2_score(y_test_real, y_pred_real)

        print(f"\n📈 Métricas do Modelo:")
        print(f"    RMSE: R$ {rmse:.2f}")
        print(f"    MAE: R$ {mae:.2f}")
        print(f"    R²: {r2:.4f}")

        # Salvar modelo e scaler
        print("💾 Salvando modelo e scaler...")
        modelo.save(f'models/{ticker}_model.h5')
        joblib.dump(scaler, f'scalers/{ticker}_scaler.pkl')

        # Salvar métricas
        metricas = {
            'ticker': ticker,
            'nome': nome_ticker,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'data_treino': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'janela': janela,
            'features': features
        }
        joblib.dump(metricas, f'metrics/{ticker}_metrics.pkl')

        # Visualização
        plt.figure(figsize=(14, 6)) # Aumentado o tamanho
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Treino Loss')
        plt.plot(history.history['val_loss'], label='Validação Loss')
        if 'mae' in history.history and 'val_mae' in history.history:
            plt.plot(history.history['mae'], label='Treino MAE', linestyle='--')
            plt.plot(history.history['val_mae'], label='Validação MAE', linestyle='--')
        plt.title(f'Curvas de Aprendizado - {nome_ticker}')
        plt.xlabel('Épocas')
        plt.ylabel('Erro (Loss/MAE)')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        n_points = min(100, len(y_test_real)) # Aumentado para 100 pontos
        plt.plot(y_test_real[-n_points:], label='Real', linewidth=2, marker='o', markersize=4)
        plt.plot(y_pred_real[-n_points:], label='Previsto', linewidth=2, alpha=0.8, marker='x', markersize=4)
        plt.title(f'Previsão vs Real (Últimos {n_points} dias) - {nome_ticker}')
        plt.xlabel(f'Dias (Últimos {n_points})')
        plt.ylabel('Preço (R$)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print(f"✅ Modelo para {nome_ticker} treinado com sucesso!")
        return metricas

    except Exception as e:
        print(f"❌ Erro ao treinar {ticker}: {str(e)}")
        import traceback
        traceback.print_exc() # Imprime o traceback completo para depuração
        return None

# === 9. Executar Treinamento ===
print("🚀 INICIANDO TREINAMENTO DOS MODELOS")
print(f"📅 Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📊 Total de tickers: {len(TICKERS)}")

resultados = []
for ticker, nome in TICKERS.items():
    resultado = treinar_modelo_ticker(ticker, nome)
    if resultado:
        resultados.append(resultado)

# === 10. Resumo Final ===
print("\n" + "="*60)
print("📊 RESUMO DO TREINAMENTO")
print("="*60)
if resultados:
    print(f"✅ Modelos treinados com sucesso: {len(resultados)}/{len(TICKERS)}")
    print("\nMelhores modelos por R² (top 5):")
    resultados_ordenados = sorted(resultados, key=lambda x: x['r2'], reverse=True)
    for i, res in enumerate(resultados_ordenados[:5]):
        print(f"{i+1}. {res['nome']} ({res['ticker']}): R² = {res['r2']:.4f}, RMSE = R$ {res['rmse']:.2f}, MAE = R$ {res['mae']:.2f}")

    if len(resultados) < len(TICKERS):
        print("\n⚠️ Alguns modelos não foram treinados. Verifique os logs acima.")
else:
    print("\n⚠️ Nenhum modelo foi treinado com sucesso. Verifique os logs acima.")

print("\n✅ Treinamento concluído!")
print("📁 Modelos salvos em: ./models/")
print("📁 Scalers salvos em: ./scalers/")
print("📁 Métricas salvas em: ./metrics/")
