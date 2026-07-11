# CLAUDE.md

Orientações para o Claude Code ao trabalhar neste repositório.

## Visão geral

Aplicação **web local em Streamlit** que aplica modelos de Machine Learning sobre
dados históricos de ações da bolsa brasileira (B3 / IBOV) para prever se o ativo
tende a **subir num horizonte futuro configurável** (1 a 20 pregões) e transforma
essa previsão numa recomendação **"Compre!" / "Não Compre!"**, validada por
walk-forward e acompanhada de um backtest simples vs. comprar-e-manter.

Projeto acadêmico (TCC de MBA em Ciência de Dados para o Mercado Financeiro).
Originalmente escrito em Dash (`app.py`, não versionado); a versão atual é a
refatoração para Streamlit. Uso local (rodado via `streamlit run`).

Idioma do projeto: **português** (código, comentários, UI e docs). Mantenha esse
padrão ao editar.

## Como rodar

```powershell
# primeira vez
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# executar (abre em http://localhost:8501)
streamlit run streamlit_app.py
```

- **Python 3.12+**, ambiente primário Windows / PowerShell.
- Requer **internet**: os preços são baixados do Yahoo Finance a cada execução.
- `TA-Lib` normalmente instala via wheel do PyPI no Windows (sem compilar o C).

## Estrutura

```
streamlit_app.py          # TODA a aplicação (UI + pipeline de ML) — arquivo único
parameters.json           # hiperparâmetros de cada modelo sklearn
requirements.txt          # dependências
dataset/                  # CSV (sep=";") com Ticker;Company_Name dos ativos do IBOV
assets/cerulean.css       # tema visual (cor #2fa4e7, herdada da versão Dash)
.venv/                    # ambiente virtual (gitignored)
```

## Arquitetura de `streamlit_app.py`

Script Streamlit linear (top-to-bottom, re-executado a cada interação):

- **Constantes/dicionários** no topo: `VARIABLE_OPTIONS` (label → nome da coluna
  do indicador, incluindo `Ibov_Retorno`), `MODEL_OPTIONS` (label → chave curta
  `lr`/`dt`/`rf`/`mlp`/`xgb`) e os dicts `*_HELP` que alimentam a seção "Help".
- **Funções com cache** (`@st.cache_data`): `load_ticker_names`,
  `load_parameters`, `download_prices(ticker, year, suffix=".SA")` — o `suffix`
  vazio é usado para baixar o índice Ibovespa (`^BVSP`).
- **`build_features(df_prices, horizon=1, threshold=0.0)`**: calcula indicadores
  TA-Lib + a coluna `Alvo`, com base no retorno cumulativo dos próximos `horizon`
  pregões. Se `threshold` (%) > 0, cria uma "zona morta" (`Alvo=NaN`) para retornos
  futuros pequenos. As últimas `horizon` linhas também ficam com `Alvo=NaN` (sem
  retorno futuro conhecido) — de propósito, para sustentar a predição "ao vivo".
  Padrões de candle são recodificados de 100/-100/0 para +1/-1/0.
- **`build_model(model_choiced, params, x_train, y_train)`**: instancia e treina
  o modelo escolhido a partir de `parameters.json`.
- **`run_walk_forward(...)`**: valida via `TimeSeriesSplit` (janela expansiva,
  `gap=horizon`) — cada fold ajusta seu próprio scaler/modelo; retorna métricas por
  fold e as predições out-of-sample concatenadas (usadas no backtest).
- **`run_backtest(oos_df, df_features, confidence_threshold)`**: simula uma
  estratégia (comprado quando sinal ≥ limiar de confiança, senão caixa) vs.
  comprar-e-manter, usando o retorno real t→t+1 da série de preços contínua.
- **Bloco de UI + pipeline**: sidebar (ativo, ano de download, horizonte, limiar de
  magnitude, variáveis, modelo, nº de folds, limiar de confiança) → validações →
  download (ativo + Ibovespa se selecionado) → features → `df_labeled` (rotulado)
  + `live_row` (linha mais recente, sem rótulo) → walk-forward → modelo de produção
  (treinado em `df_labeled` completo) → gráfico de preço + métricas agregadas +
  backtest + recomendação final (via `predict_proba` do modelo de produção).

## Detalhes importantes (não quebrar)

- **Sem vazamento de dados**: dentro de cada fold do walk-forward E no modelo de
  produção, o `MinMaxScaler` é `fit` só no treino (ou em todo o `df_labeled`, no
  caso do modelo de produção) e `transform` no teste/na predição ao vivo. Nunca
  reaproveite o scaler/modelo de uma iteração do loop de folds para outra coisa.
- **`Alvo`/`RetornoFuturo` podem ser `NaN` de propósito** (cauda sem futuro válido
  + zona morta do limiar) — o `dropna` final de `build_features` filtra apenas por
  colunas de indicador (`indicator_cols`), nunca por essas duas. Um `dropna()` sem
  `subset` apagaria a linha usada para a recomendação "ao vivo".
- **`random_state: 42`** em todos os modelos de `parameters.json` (resultados
  reprodutíveis). Mantenha ao editar hiperparâmetros.
- **`yfinance` usa `period="max"`** de propósito: versões recentes retornam só
  ~1 mês quando o período não é informado, o que zerava os indicadores longos
  (MM200 etc.). Não troque por datas soltas sem testar.
- **Colunas MultiIndex do yfinance**: `download_prices` já achata (`get_level_values(0)`)
  o retorno de ticker único.
- **MLP**: `hidden_layer_sizes` vem como lista do JSON e precisa ser convertido
  para `tuple` antes de instanciar o `MLPClassifier` (já tratado em `build_model`).
- Em `parameters.json` cada modelo é uma **lista com um único dict** de config;
  o código acessa `[0]`.
- **`gap` do `TimeSeriesSplit` é em nº de linhas, não em dias de calendário** —
  aproximação aceita quando o limiar de magnitude torna o dataset rotulado esparso.
  Documentado como limitação conhecida, não vale a pena um splitter customizado.
- Caminhos são resolvidos via `BASE_DIR` (relativo ao arquivo). Ao adicionar
  arquivos de dados, use o mesmo padrão.

## Convenções

- Não há suíte de testes nem linter configurado. Valide mudanças rodando o app
  (`streamlit run streamlit_app.py`) e exercitando a interface.
- Toda a lógica vive em `streamlit_app.py` — mantenha o app coeso ali salvo motivo
  forte para dividir.

## Aviso de escopo

É uma **estimativa estatística de curtíssimo prazo** baseada só em indicadores
técnicos, para fins acadêmicos. Não é recomendação formal de investimento — não
adicione linguagem que sugira garantia de retorno.
