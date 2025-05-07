# Projeto de Previsão de Corrosão em FPSO

Elison Farias Olveira - UFRJ/COPPE - elisonmartelo@gmail.com - 2025 

Este projeto implementa um sistema de previsão de corrosão para FPSOs (Floating Production Storage and Offloading) utilizando ML.NET. O sistema é capaz de prever dois parâmetros de corrosão (C1 e C2) com base em características estruturais e operacionais do FPSO.

## Estrutura do Projeto

O projeto está organizado da seguinte forma:

- `Program.cs`: Arquivo principal contendo a implementação do sistema
- `dadosFPSO.csv`: Arquivo de entrada com dados históricos para treinamento
- `dados_para_prever.csv`: Arquivo de entrada com dados para previsão
- Arquivos gerados durante a execução:
  - `dados_treino.csv`: 70% dos dados para treinamento
  - `dados_validacao.csv`: 15% dos dados para validação
  - `dados_teste.csv`: 15% dos dados para teste
  - `metricas_treinamento.csv`: Métricas de desempenho dos modelos
  - `previsoes.csv`: Resultados das previsões
  - `model_c1.zip` e `model_c2.zip`: Modelos treinados

## Requisitos

- .NET 9.0 ou superior
- Arquivos de entrada no formato CSV com separador ';'
- Valores decimais usando vírgula como separador

## Como Usar

### Preparação dos Dados

1. Prepare o arquivo `dadosFPSO.csv` com os dados históricos no seguinte formato:
   ```
   PosicaoLongitudinal;SecaoCargaInterface;Elemento;Altura;AnosVida;AnosCoat;Flag;C1;C2
   ```

2. Prepare o arquivo `dados_para_prever.csv` com os dados para previsão:
   ```
   PosicaoLongitudinal;SecaoCargaInterface;Elemento;Altura;AnosVida;AnosCoat;Flag
   ```

### Execução

1. Abra o terminal (CMD) e navegue até a pasta do projeto:
   ```
   cd caminho/para/CorrosaoFPSO_New
   ```

2. Para treinar os modelos:
   ```
   dotnet run train
   ```
   Este comando irá:
   - Converter os dados para formato internacional
   - Separar os dados em treino, validação e teste
   - Treinar os modelos C1 e C2
   - Salvar as métricas em `metricas_treinamento.csv`
   - Salvar os modelos em `model_c1.zip` e `model_c2.zip`

3. Para fazer previsões:
   ```
   dotnet run predict
   ```
   Este comando irá:
   - Carregar os modelos treinados
   - Fazer previsões para os dados em `dados_para_prever.csv`
   - Salvar os resultados em `previsoes.csv`

## Implementação

### Classes de Dados

- `DadosTreino`: Classe para dados de treinamento com C1 e C2
- `DadosPrevisao`: Classe para dados de previsão (sem C1 e C2)
- `Resultado`: Classe para armazenar resultados das previsões

### Pipeline de ML

O sistema utiliza dois pipelines separados:

1. Pipeline para C1:
   - Features: PosicaoLongitudinal, Altura, AnosVida, AnosCoat
   - Algoritmo: FastForest

2. Pipeline para C2:
   - Features: PosicaoLongitudinal, Altura, AnosVida, AnosCoat
   - Algoritmo: FastForest

### Divisão dos Dados

Os dados são divididos em três conjuntos:
- Treino: 70% dos dados
- Validação: 15% dos dados
- Teste: 15% dos dados

### Métricas

O sistema avalia os modelos usando:
- R² Score: Mede a qualidade do ajuste do modelo
- RMSE (Root Mean Square Error): Mede o erro médio das previsões

## Resultados

Os resultados são salvos em dois arquivos:

1. `metricas_treinamento.csv`:
   - Métricas de desempenho para modelos C1 e C2
   - Inclui R² Score e RMSE

2. `previsoes.csv`:
   - Dados originais
   - Colunas C1_Previsto e C2_Previsto com os valores previstos 
