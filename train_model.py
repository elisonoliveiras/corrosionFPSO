import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import backend as K

def print_debug(message):
    print(message)
    sys.stdout.flush()

print_debug("Iniciando o script...")

# Carregar os dados
print_debug("Carregando dados...")
try:
    # Tentar diferentes codificações
    encodings = ['latin1', 'utf-8', 'cp1252']
    df = None
    
    for encoding in encodings:
        try:
            print_debug(f"Tentando codificação: {encoding}")
            df = pd.read_csv('dadosFPSO.csv', sep=';', encoding=encoding, decimal=',')
            print_debug(f"Sucesso com codificação: {encoding}")
            break
        except Exception as e:
            print_debug(f"Falha com codificação {encoding}: {str(e)}")
            continue
    
    if df is None:
        raise Exception("Não foi possível ler o arquivo com nenhuma codificação")
    
    print_debug(f"Arquivo carregado com sucesso. Shape: {df.shape}")
    print_debug("Colunas disponíveis: " + str(df.columns.tolist()))
    print_debug("Tipos de dados: " + str(df.dtypes))
    
    # Verificar valores nulos
    print_debug("\nVerificando valores nulos:")
    print_debug(df.isnull().sum())
    
    # Verificar valores únicos em cada coluna
    print_debug("\nValores únicos em cada coluna:")
    for col in df.columns:
        print_debug(f"{col}: {len(df[col].unique())} valores únicos")
    
    # Após carregar o DataFrame
    print_debug(f"Primeiros valores de C1 no DataFrame: {df['C1'].head(10).tolist()}")
    print_debug(f"Valores únicos de C1: {np.unique(df['C1'])[:20]}")
    print_debug(f"Total de linhas com C1=0: {(df['C1'] == 0).sum()}")
    
except Exception as e:
    print_debug(f"Erro ao carregar o arquivo: {e}")
    exit(1)

# Converter valores categóricos
print_debug("\nConvertendo valores categóricos...")
try:
    encoders = {}
    categorical_columns = ['seção_carga_interface', 'elemento']
    
    for col in categorical_columns:
        if df[col].dtype == 'object':
            print_debug(f"Convertendo coluna {col}...")
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col]).astype(np.float32)
            print_debug(f"Coluna {col} convertida. Valores únicos: {len(encoders[col].classes_)}")
            print_debug(f"Categorias de {col}: {list(encoders[col].classes_)}")
    
    # Converter flag para binário
    df['flag'] = df['flag'].map({'Sim': 1, 'Não': 0}).astype(np.float32)
    
    print_debug("Conversão concluída com sucesso")
except Exception as e:
    print_debug(f"Erro ao converter valores categóricos: {e}")
    exit(1)

# Separar features e targets
print_debug("\nPreparando features e targets...")
try:
    X = df[['posição_longitudinal', 'seção_carga_interface', 'elemento', 'altura', 'anos_vida', 'anos_coat', 'flag']].values
    y = df[['C1', 'C2']].values
    print_debug(f"Features shape: {X.shape}")
    print_debug(f"Targets shape: {y.shape}")
    
    # Após separar features e targets
    print_debug(f"Primeiros valores de y (targets): {y[:10, 0]}")
    
except Exception as e:
    print_debug(f"Erro ao preparar features e targets: {e}")
    exit(1)

# Dividir em treino, validação e teste
print_debug("\nDividindo dados em treino, validação e teste...")
try:
    # Primeiro split: 95% treino, 5% restante
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.05, random_state=42)
    # Segundo split: 50% validação, 50% teste (do 5% restante)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print_debug(f"Treino shape: {X_train.shape}")
    print_debug(f"Validação shape: {X_val.shape}")
    print_debug(f"Teste shape: {X_test.shape}")
    
    # Antes de salvar os conjuntos de dados, fazer o desencoding das colunas categóricas
    for split_name, X_split, y_split in [
        ("dados_treino.csv", X_train, y_train),
        ("dados_validacao.csv", X_val, y_val),
        ("dados_teste.csv", X_test, y_test)
    ]:
        df_split = pd.DataFrame(X_split, columns=['posição_longitudinal', 'seção_carga_interface', 'elemento', 'altura', 'anos_vida', 'anos_coat', 'flag'])
        # Desencoding das colunas categóricas
        if 'seção_carga_interface' in encoders:
            df_split['seção_carga_interface'] = encoders['seção_carga_interface'].inverse_transform(df_split['seção_carga_interface'].astype(int))
        if 'elemento' in encoders:
            df_split['elemento'] = encoders['elemento'].inverse_transform(df_split['elemento'].astype(int))
        # Adicionar as colunas alvo
        df_split['C1'] = y_split[:, 0]
        df_split['C2'] = y_split[:, 1]
        df_split.to_csv(split_name, sep=';', index=False, decimal=',')
except Exception as e:
    print_debug(f"Erro ao dividir dados: {e}")
    exit(1)

# Normalizar os dados
print_debug("\nNormalizando dados...")
try:
    # Separar features numéricas (sem a flag)
    numeric_columns = ['posição_longitudinal', 'altura', 'anos_vida', 'anos_coat']
    categorical_columns = ['seção_carga_interface', 'elemento']
    binary_columns = ['flag']
    
    # Criar MinMaxScaler para features numéricas
    scaler = MinMaxScaler()
    
    # Índices das colunas numéricas (sem a flag)
    # posição_longitudinal: 0, altura: 3, anos_vida: 4, anos_coat: 5
    X_train_numeric = X_train[:, [0, 3, 4, 5]]
    X_val_numeric = X_val[:, [0, 3, 4, 5]]
    X_test_numeric = X_test[:, [0, 3, 4, 5]]
    
    X_train_scaled_numeric = scaler.fit_transform(X_train_numeric)
    X_val_scaled_numeric = scaler.transform(X_val_numeric)
    X_test_scaled_numeric = scaler.transform(X_test_numeric)
    
    # Categóricas
    X_train_cat = X_train[:, [1, 2]]
    X_val_cat = X_val[:, [1, 2]]
    X_test_cat = X_test[:, [1, 2]]
    
    # Flag (binária) - NÃO normalizar
    X_train_bin = X_train[:, 6].reshape(-1, 1)
    X_val_bin = X_val[:, 6].reshape(-1, 1)
    X_test_bin = X_test[:, 6].reshape(-1, 1)
    
    # Concatenar: numéricas normalizadas + categóricas codificadas + binária (flag)
    X_train_scaled = np.hstack((X_train_scaled_numeric, X_train_cat, X_train_bin))
    X_val_scaled = np.hstack((X_val_scaled_numeric, X_val_cat, X_val_bin))
    X_test_scaled = np.hstack((X_test_scaled_numeric, X_test_cat, X_test_bin))
    
    # Log dos valores únicos para debug
    for i in range(X_train.shape[1]):
        print_debug(f"Feature {i} - Valores únicos: {len(np.unique(X_train[:, i]))}")
        print_debug(f"Feature {i} - Min: {np.min(X_train[:, i])}, Max: {np.max(X_train[:, i])}")
        # Só mostrar valores normalizados para as features numéricas (índices 0, 3, 4, 5)
        if i in [0, 3, 4, 5]:
            idx_num = [0, 3, 4, 5].index(i)
            print_debug(f"Feature {i} (normalizada) - Min: {np.min(X_train_scaled_numeric[:, idx_num])}, Max: {np.max(X_train_scaled_numeric[:, idx_num])}")
    
    # Após normalização das features
    print_debug("\nEstatísticas das features normalizadas (treino):")
    for i in range(X_train_scaled.shape[1]):
        col = X_train_scaled[:, i]
        print_debug(f"Feature {i}: min={np.min(col)}, max={np.max(col)}, mean={np.mean(col)}")
    
    print_debug("Dados normalizados com sucesso")
except Exception as e:
    print_debug(f"Erro ao normalizar dados: {e}")
    exit(1)

# Verificar valores NaN
print_debug("\nVerificando valores NaN...")
try:
    nan_check_train = np.isnan(X_train_scaled).sum()
    nan_check_val = np.isnan(X_val_scaled).sum()
    nan_check_test = np.isnan(X_test_scaled).sum()
    
    print_debug(f"NaN em X_train: {nan_check_train}")
    print_debug(f"NaN em X_val: {nan_check_val}")
    print_debug(f"NaN em X_test: {nan_check_test}")
    
    if nan_check_train > 0 or nan_check_val > 0 or nan_check_test > 0:
        print_debug("AVISO: Existem valores NaN nos dados!")
        # Substituir NaN pela média
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
        X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)
except Exception as e:
    print_debug(f"Erro ao verificar NaN: {e}")
    exit(1)

# Criar o modelo
print_debug("\nCriando modelo...")
try:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(7,), name='input'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='linear', name='output')
    ])
    print_debug("Modelo criado com sucesso")
except Exception as e:
    print_debug(f"Erro ao criar modelo: {e}")
    exit(1)

# Função de perda customizada para mascarar C1 quando C1 == 0
def custom_loss(y_true, y_pred):
    mse_c1 = tf.square(y_true[:, 0] - y_pred[:, 0])
    mse_c2 = tf.square(y_true[:, 1] - y_pred[:, 1])
    loss_c1 = tf.reduce_mean(mse_c1)
    loss_c2 = tf.reduce_mean(mse_c2)
    return loss_c1 + loss_c2

# Compilar o modelo
print_debug("\nCompilando modelo...")
try:
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,  # menor para mais estabilidade
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    model.compile(
        optimizer=optimizer,
        loss=custom_loss,
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
    )
    print_debug("Modelo compilado com sucesso")
except Exception as e:
    print_debug(f"Erro ao compilar modelo: {e}")
    exit(1)

# Ajustar early stopping e redução de learning rate
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,  # mais paciência para redes mais profundas
    restore_best_weights=True,
    min_delta=1e-5
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-6,
    verbose=1
)

# Criar arquivo para métricas
metrics_file = open('metricas.csv', 'w', newline='')
metrics_writer = csv.writer(metrics_file)
metrics_writer.writerow(['epoch', 'train_mse_c1', 'train_mse_c2', 'train_rmse_c1', 'train_rmse_c2', 
                        'train_r2_c1', 'train_r2_c2', 'val_mse_c1', 'val_mse_c2', 'val_rmse_c1', 
                        'val_rmse_c2', 'val_r2_c1', 'val_r2_c2'])

# Callback personalizado para salvar métricas
class MetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Predições para treino e validação
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        
        # Métricas para C1
        train_mse_c1 = mean_squared_error(y_train[:, 0], y_train_pred[:, 0])
        train_rmse_c1 = np.sqrt(train_mse_c1)
        train_r2_c1 = r2_score(y_train[:, 0], y_train_pred[:, 0])
        
        val_mse_c1 = mean_squared_error(y_val[:, 0], y_val_pred[:, 0])
        val_rmse_c1 = np.sqrt(val_mse_c1)
        val_r2_c1 = r2_score(y_val[:, 0], y_val_pred[:, 0])
        
        # Métricas para C2
        train_mse_c2 = mean_squared_error(y_train[:, 1], y_train_pred[:, 1])
        train_rmse_c2 = np.sqrt(train_mse_c2)
        train_r2_c2 = r2_score(y_train[:, 1], y_train_pred[:, 1])
        
        val_mse_c2 = mean_squared_error(y_val[:, 1], y_val_pred[:, 1])
        val_rmse_c2 = np.sqrt(val_mse_c2)
        val_r2_c2 = r2_score(y_val[:, 1], y_val_pred[:, 1])
        
        # Escrever métricas no arquivo
        metrics_writer.writerow([
            epoch + 1,
            train_mse_c1, train_mse_c2,
            train_rmse_c1, train_rmse_c2,
            train_r2_c1, train_r2_c2,
            val_mse_c1, val_mse_c2,
            val_rmse_c1, val_rmse_c2,
            val_r2_c1, val_r2_c2
        ])
        metrics_file.flush()
        
        # Adicionar histograma das previsões
        if epoch % 10 == 0:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(2, 2, 1)
            plt.hist(y_train[:, 0], bins=50, alpha=0.5, label='Real C1')
            plt.hist(y_train_pred[:, 0], bins=50, alpha=0.5, label='Previsto C1')
            plt.legend()
            plt.title(f'Distribuição C1 - Epoch {epoch+1}')
            
            plt.subplot(2, 2, 2)
            plt.hist(y_train[:, 1], bins=50, alpha=0.5, label='Real C2')
            plt.hist(y_train_pred[:, 1], bins=50, alpha=0.5, label='Previsto C2')
            plt.legend()
            plt.title(f'Distribuição C2 - Epoch {epoch+1}')
            
            plt.subplot(2, 2, 3)
            plt.scatter(y_train[:, 0], y_train_pred[:, 0], alpha=0.1)
            plt.plot([y_train[:, 0].min(), y_train[:, 0].max()], 
                    [y_train[:, 0].min(), y_train[:, 0].max()], 'r--')
            plt.xlabel('Real C1')
            plt.ylabel('Previsto C1')
            
            plt.subplot(2, 2, 4)
            plt.scatter(y_train[:, 1], y_train_pred[:, 1], alpha=0.1)
            plt.plot([y_train[:, 1].min(), y_train[:, 1].max()],
                    [y_train[:, 1].min(), y_train[:, 1].max()], 'r--')
            plt.xlabel('Real C2')
            plt.ylabel('Previsto C2')
            
            plt.tight_layout()
            plt.savefig(f'distribuicao_epoch_{epoch+1}.png')
            plt.close()

# Criar máscara para C1: 1 onde C1 != 0, 0 onde C1 == 0
sample_weight_c1 = (y_train[:, 0] != 0).astype(np.float32)
sample_weight_c2 = np.ones_like(y_train[:, 1], dtype=np.float32)
# Peso total: média dos dois alvos
sample_weight = np.stack([sample_weight_c1, sample_weight_c2], axis=1)

# Treinar o modelo
print_debug("\nTreinando modelo...")
try:
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=64,
        validation_data=(X_val_scaled, y_val),
        sample_weight=sample_weight,
        callbacks=[early_stopping, reduce_lr, MetricsCallback()],
        verbose=1
    )
    print_debug("Modelo treinado com sucesso")
except Exception as e:
    print_debug(f"Erro ao treinar modelo: {e}")
    exit(1)

# Avaliar o modelo no conjunto de teste
print_debug("\nAvaliando modelo no conjunto de teste...")
try:
    y_test_pred = model.predict(X_test_scaled)
    
    test_mse_c1 = mean_squared_error(y_test[:, 0], y_test_pred[:, 0])
    test_rmse_c1 = np.sqrt(test_mse_c1)
    test_r2_c1 = r2_score(y_test[:, 0], y_test_pred[:, 0])
    
    test_mse_c2 = mean_squared_error(y_test[:, 1], y_test_pred[:, 1])
    test_rmse_c2 = np.sqrt(test_mse_c2)
    test_r2_c2 = r2_score(y_test[:, 1], y_test_pred[:, 1])
    
    print_debug(f"Teste MSE C1: {test_mse_c1:.4f}")
    print_debug(f"Teste RMSE C1: {test_rmse_c1:.4f}")
    print_debug(f"Teste R² C1: {test_r2_c1:.4f}")
    print_debug(f"Teste MSE C2: {test_mse_c2:.4f}")
    print_debug(f"Teste RMSE C2: {test_rmse_c2:.4f}")
    print_debug(f"Teste R² C2: {test_r2_c2:.4f}")
    
    # Após predict no conjunto de teste
    print_debug(f"Primeiras previsões de C1 no teste: {y_test_pred[:10, 0]}")
except Exception as e:
    print_debug(f"Erro ao avaliar modelo: {e}")
    exit(1)

# Salvar o modelo
print_debug("\nSalvando modelo...")
try:
    # Salvar no formato SavedModel
    tf.saved_model.save(model, 'saved_model')
    print_debug(f"Data/hora do modelo salvo: {os.path.getmtime('saved_model')}")
    print_debug("Modelo salvo com sucesso")
except Exception as e:
    print_debug(f"Erro ao salvar modelo: {e}")
    exit(1)

# Salvar o scaler e os encoders
print_debug("\nSalvando scaler e encoders...")
try:
    import joblib
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(encoders, 'encoders.pkl')
    print_debug("Scaler e encoders salvos com sucesso")
except Exception as e:
    print_debug(f"Erro ao salvar scaler e encoders: {e}")
    exit(1)

# Fechar arquivo de métricas
metrics_file.close()

# Após o treinamento, plotar histórico de treinamento
print_debug("\nGerando gráficos de histórico...")
try:
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Loss (MSE)')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(history.history['mae'], label='Treino')
    plt.plot(history.history['val_mae'], label='Validação')
    plt.title('MAE')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(history.history['root_mean_squared_error'], label='Treino')
    plt.plot(history.history['val_root_mean_squared_error'], label='Validação')
    plt.title('RMSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('historico_treinamento.png')
    plt.close()
    
    # Análise de correlação entre features
    plt.figure(figsize=(10, 8))
    correlation_matrix = pd.DataFrame(X_train).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlação entre Features')
    plt.savefig('correlacao_features.png')
    plt.close()
    
    print_debug("Gráficos salvos com sucesso")
except Exception as e:
    print_debug(f"Erro ao gerar gráficos: {e}")

print_debug("\nProcesso concluído com sucesso!") 