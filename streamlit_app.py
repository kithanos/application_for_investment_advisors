# Aplicação de ML para Assessores de Investimento - versão Streamlit
# Refatoração do dashboard originalmente escrito em Dash (app.py)

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import talib
import yfinance as yf

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from xgboost import XGBClassifier

import warnings

warnings.filterwarnings("ignore")

# Caminhos resolvidos relativamente a este arquivo, para funcionar tanto ao rodar
# via `streamlit run` quanto empacotado com o PyInstaller (.exe).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILENAME = os.path.join(BASE_DIR, "dataset", "Brazillian_Ticker_Codes_IBOV.csv")
PARAMETERS_FILENAME = os.path.join(BASE_DIR, "parameters.json")

# Cor tema (cerulean) herdada da versão Dash
THEME_COLOR = "#2fa4e7"

# Variáveis (features) disponíveis para o modelo
VARIABLE_OPTIONS = {
    "Volatilidade": "Volatilidade",
    "Relative Strength Index": "RSI",
    "Simple Moving Average 50 days": "MM50",
    "Simple Moving Average 200 days": "MM200",
    "Doji Star": "Doji Star",
    "Hagingman": "Hagingman",
    "Average Directional Movement Index": "ADX",
    "Kaufman Adaptive Moving Average 50 days": "KAMA50",
    "Kaufman Adaptive Moving Average 200 days": "KAMA200",
    "Weighted Moving Average 50 days": "WMA50",
    "Weighted Moving Average 200 days": "WMA200",
    "Average Directional Movement Index Rating": "ADXR",
    "Rate of change": "ROC",
    "Hilbert Transform - Dominant Cycle Period": "HT_DCPERIOD",
    "Three Inside Up/Down": "CDL3INSIDE",
    "Rickshaw Man": "CDLRICKSHAWMAN",
    "Retorno do Ibovespa": "Ibov_Retorno",
}

MODEL_OPTIONS = {
    "Logistic Regression": "lr",
    "Decision Tree": "dt",
    "Random Forest": "rf",
    "Rede Neural": "mlp",
    "XGBoost": "xgb",
}

# Descrições usadas na seção "Help" (chaves alinhadas a VARIABLE_OPTIONS/MODEL_OPTIONS)
VARIABLE_HELP = {
    "Volatilidade": "Desvio-padrão dos retornos dos últimos 20 pregões, anualizado. "
    "Mede o quanto o preço oscila — quanto maior, mais arriscado/instável o ativo.",
    "Relative Strength Index": "RSI (14 períodos). Oscilador de 0 a 100 que indica se o ativo "
    "está sobrecomprado (acima de ~70) ou sobrevendido (abaixo de ~30).",
    "Simple Moving Average 50 days": "Média móvel simples do preço nos últimos 50 pregões. "
    "Suaviza o preço e mostra a tendência de médio prazo.",
    "Simple Moving Average 200 days": "Média móvel simples do preço nos últimos 200 pregões. "
    "Referência clássica de tendência de longo prazo.",
    "Doji Star": "Padrão de candle (vela) de reversão. Codificado como +1 (alta), -1 (baixa) "
    "ou 0 (ausente).",
    "Hagingman": "Padrão de candle 'Hanging Man' (enforcado), sinal de possível reversão de "
    "alta para baixa. Codificado como +1, -1 ou 0.",
    "Average Directional Movement Index": "ADX (14 períodos). Mede a FORÇA da tendência "
    "(não a direção) — valores altos indicam tendência forte.",
    "Kaufman Adaptive Moving Average 50 days": "KAMA (50 períodos). Média móvel que se adapta à "
    "volatilidade: reage rápido em tendência e filtra ruído em mercado lateral.",
    "Kaufman Adaptive Moving Average 200 days": "KAMA (200 períodos). Versão de longo prazo da "
    "média móvel adaptativa de Kaufman.",
    "Weighted Moving Average 50 days": "Média móvel ponderada (50 períodos), que dá mais peso aos "
    "preços mais recentes do que a média simples.",
    "Weighted Moving Average 200 days": "Média móvel ponderada de longo prazo (200 períodos).",
    "Average Directional Movement Index Rating": "ADXR. Versão suavizada do ADX, usada para "
    "confirmar a força da tendência.",
    "Rate of change": "ROC (14 períodos). Variação percentual do preço em relação a 14 pregões "
    "atrás — um indicador de momento (momentum).",
    "Hilbert Transform - Dominant Cycle Period": "Estima o comprimento do ciclo dominante do "
    "preço usando a Transformada de Hilbert.",
    "Three Inside Up/Down": "Padrão de candle de reversão de três velas. Codificado como +1, -1 "
    "ou 0.",
    "Rickshaw Man": "Padrão de candle de indecisão (corpo pequeno, sombras longas). Codificado "
    "como +1, -1 ou 0.",
    "Retorno do Ibovespa": "Retorno diário do índice Ibovespa (^BVSP), já conhecido no fechamento "
    "do mesmo pregão usado para prever o dia seguinte. Captura o efeito do mercado como um todo "
    "sobre o ativo.",
}

MODEL_HELP = {
    "Logistic Regression": "Regressão Logística — modelo linear que estima a probabilidade de o "
    "ativo subir. Simples, rápido e fácil de interpretar; bom ponto de partida.",
    "Decision Tree": "Árvore de Decisão — sequência de perguntas do tipo 'se/então' sobre os "
    "indicadores. Fácil de visualizar, mas sozinha tende a decorar os dados (overfitting).",
    "Random Forest": "Floresta Aleatória — combina muitas árvores de decisão e faz uma votação. "
    "Geralmente mais precisa e estável do que uma árvore única.",
    "Rede Neural": "Rede Neural (MLP) — rede com camadas de neurônios capaz de captar relações "
    "não-lineares complexas. Mais poderosa, porém exige mais dados e é menos interpretável.",
    "XGBoost": "XGBoost — floresta de árvores em gradient boosting, em que cada árvore nova "
    "corrige os erros das anteriores. Costuma superar a Random Forest em dados tabulares, à "
    "custa de mais hiperparâmetros e menor interpretabilidade.",
}

METRIC_HELP = {
    "Recall": "Dos pregões que realmente subiram, qual fração o modelo acertou. Recall alto = "
    "perde poucas altas.",
    "Precision": "Das vezes que o modelo disse 'Compre!', qual fração de fato subiu. Precision "
    "alta = poucos alarmes falsos.",
    "f1-score": "Média harmônica entre Precision e Recall — equilíbrio entre os dois.",
    "Acurácia": "Proporção total de previsões corretas (altas e quedas) sobre o conjunto de teste.",
    "Limiar de confiança": "Probabilidade mínima que o modelo precisa prever para o movimento ser "
    "considerado 'Compre!'. Usado de forma consistente nas métricas por fold, no backtest e na "
    "recomendação final.",
    "Retorno da estratégia": "Retorno acumulado simulado ao seguir os sinais do modelo (comprado "
    "quando o sinal indica alta, em caixa nos demais dias) no período avaliado pelo walk-forward.",
    "Retorno comprar e manter": "Retorno acumulado de simplesmente manter o ativo comprado durante "
    "todo o mesmo período, como referência de comparação para a estratégia.",
    "Sharpe simplificado": "Retorno médio dividido pelo desvio-padrão dos retornos diários, "
    "anualizado (×√252), sem descontar uma taxa livre de risco. Mede o retorno obtido por unidade "
    "de risco — quanto maior, melhor.",
}


@st.cache_data
def load_ticker_names():
    """Lê o dataset com o código (ticker) e o nome do ativo."""
    df = pd.read_csv(DATASET_FILENAME, sep=";")
    # O CSV tem um espaço após o ";" (ex.: "BBAS3; Banco do Brasil S.A."), então
    # Company_Name vem com espaço à esquerda — sem o strip, o nome combinado ficava
    # com espaço duplo e nunca coincidia com o `default_stock` da sidebar.
    df["Ticker"] = df["Ticker"].str.strip()
    df["Company_Name"] = df["Company_Name"].str.strip()
    df["Ticker_CompanyNames"] = df["Ticker"] + " - " + df["Company_Name"]
    return df


@st.cache_data
def load_parameters():
    """Carrega os hiperparâmetros dos modelos do parameters.json."""
    with open(PARAMETERS_FILENAME) as f:
        return json.load(f)


@st.cache_data(show_spinner="Baixando dados do ativo...")
def download_prices(ticker, year, suffix=".SA"):
    """Baixa o histórico de preços do ativo via Yahoo Finance.

    `suffix` é ".SA" para ativos da B3 e "" para índices como o Ibovespa
    (^BVSP), que não usam esse sufixo no Yahoo Finance.
    """
    # period="max": versões recentes do yfinance passaram a retornar só ~1 mês
    # quando período/datas não são informados, o que zerava os indicadores longos.
    df = yf.download(f"{ticker}{suffix}", period="max", auto_adjust=False, progress=False)

    # yfinance pode retornar colunas MultiIndex para um único ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Filtra os dados de acordo com o ano selecionado pelo usuário
    df = df.loc[df.index > f"{year}-12-31"]
    return df


def build_features(df_prices, horizon=1, threshold=0.0):
    """Calcula os indicadores técnicos e a variável alvo.

    O alvo usa o retorno cumulativo dos próximos `horizon` pregões. Se
    `threshold` (em %) for maior que zero, cria-se uma "zona morta": só é
    marcado Alvo=1 (subiu) ou Alvo=0 (não subiu) quando o retorno futuro
    supera o limiar em algum sentido — o meio fica como NaN e é descartado
    como amostra ambígua. As últimas `horizon` linhas também ficam com Alvo
    NaN (ainda não há retorno futuro conhecido) e são preservadas aqui de
    propósito: é o mesmo mecanismo que sustenta a predição "ao vivo" mais
    recente, calculada no script principal.
    """
    df = df_prices.copy()
    df["Retornos"] = df["Adj Close"].pct_change(1)
    df.dropna(subset=["Retornos"], inplace=True)

    df["RetornoFuturo"] = df["Adj Close"].shift(-horizon) / df["Adj Close"] - 1
    limiar = threshold / 100
    df["Alvo"] = np.select(
        [df["RetornoFuturo"] > limiar, df["RetornoFuturo"] < -limiar],
        [1, 0],
        default=np.nan,
    )

    df["Volatilidade"] = df["Retornos"].rolling(20).std() * np.sqrt(252) * 100
    df["RSI"] = talib.RSI(df["Adj Close"], timeperiod=14)
    df["MM50"] = talib.SMA(df["Adj Close"], timeperiod=50)
    df["MM200"] = talib.SMA(df["Adj Close"], timeperiod=200)

    df["Doji Star"] = talib.CDLDOJISTAR(df["Open"], df["High"], df["Low"], df["Close"])
    df["Hagingman"] = talib.CDLHANGINGMAN(df["Open"], df["High"], df["Low"], df["Close"])
    df["Doji Star"] = np.where(df["Doji Star"] == 100, 1, np.where(df["Doji Star"] == -100, -1, 0))
    df["Hagingman"] = np.where(df["Hagingman"] == 100, 1, np.where(df["Hagingman"] == -100, -1, 0))

    df["ADX"] = talib.ADX(df["High"], df["Low"], df["Close"], timeperiod=14)
    df["KAMA50"] = talib.KAMA(df["Adj Close"], timeperiod=50)
    df["KAMA200"] = talib.KAMA(df["Adj Close"], timeperiod=200)
    df["WMA50"] = talib.WMA(df["Adj Close"], timeperiod=50)
    df["WMA200"] = talib.WMA(df["Adj Close"], timeperiod=200)
    df["ADXR"] = talib.ADXR(df["High"], df["Low"], df["Close"], timeperiod=14)
    df["ROC"] = talib.ROC(df["Adj Close"], timeperiod=14)
    df["HT_DCPERIOD"] = talib.HT_DCPERIOD(df["Adj Close"])

    df["CDL3INSIDE"] = talib.CDL3INSIDE(df["Open"], df["High"], df["Low"], df["Close"])
    df["CDL3INSIDE"] = np.where(df["CDL3INSIDE"] == 100, 1, np.where(df["CDL3INSIDE"] == -100, -1, 0))
    df["CDLRICKSHAWMAN"] = talib.CDLRICKSHAWMAN(df["Open"], df["High"], df["Low"], df["Close"])
    df["CDLRICKSHAWMAN"] = np.where(
        df["CDLRICKSHAWMAN"] == 100, 1, np.where(df["CDLRICKSHAWMAN"] == -100, -1, 0)
    )

    # Só descarta por indicador indisponível (ex.: lookback do MM200/KAMA200) — nunca
    # por "Alvo"/"RetornoFuturo", que ficam NaN de propósito na cauda e na zona morta.
    indicator_cols = [col for col in VARIABLE_OPTIONS.values() if col != "Ibov_Retorno"]
    df.dropna(subset=indicator_cols, inplace=True)
    return df


def build_model(model_choiced, params, x_train, y_train):
    """Instancia o modelo escolhido com os hiperparâmetros e o treina."""
    if model_choiced == "lr":
        cfg = params["models"]["logistic_regression"][0]
        model = LogisticRegression(**cfg)
    elif model_choiced == "dt":
        cfg = params["models"]["decision_tree"][0]
        model = DecisionTreeClassifier(**cfg)
    elif model_choiced == "rf":
        cfg = params["models"]["random_forest"][0]
        model = RandomForestClassifier(**cfg)
    elif model_choiced == "mlp":
        cfg = dict(params["models"]["mlp_classifier"][0])
        cfg["hidden_layer_sizes"] = tuple(cfg["hidden_layer_sizes"])
        model = MLPClassifier(**cfg)
    elif model_choiced == "xgb":
        cfg = params["models"]["xgboost"][0]
        model = XGBClassifier(**cfg)
    else:
        raise ValueError(f"Modelo desconhecido: {model_choiced}")

    return model.fit(x_train, y_train)


def run_walk_forward(df_labeled, list_variables, model_choiced, params, n_splits, horizon, confidence_threshold):
    """Valida o modelo em janelas expansivas (walk-forward).

    Cada fold treina com todo o histórico disponível até aquele ponto e testa
    no período seguinte; o `gap` evita que o fim do treino "veja" informação
    de preço já embutida no alvo do início do teste (o alvo usa retorno até
    `horizon` pregões à frente). O gap é contado em número de linhas, não em
    dias de calendário — uma aproximação aceitável quando o limiar de
    magnitude cria uma "zona morta" que torna o dataset rotulado esparso.

    Retorna a lista de métricas por fold e o DataFrame de predições fora da
    amostra (out-of-sample) de todos os folds, concatenadas em ordem
    temporal — usado depois no backtest.
    """
    X = df_labeled[list_variables].values
    y = df_labeled["Alvo"].values
    dates = df_labeled.index

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
    fold_metrics = []
    oos_records = []

    for train_idx, test_idx in tscv.split(X):
        y_train, y_test = y[train_idx], y[test_idx]
        if len(test_idx) == 0 or len(set(y_train)) < 2:
            continue
        try:
            scaler = MinMaxScaler(feature_range=(0, 1))
            x_train = scaler.fit_transform(X[train_idx])
            x_test = scaler.transform(X[test_idx])
            model = build_model(model_choiced, params, x_train, y_train)
            y_proba = model.predict_proba(x_test)[:, 1]
        except Exception:
            continue

        y_pred = (y_proba >= confidence_threshold).astype(int)
        fold_metrics.append(
            {
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "f1-score": f1_score(y_test, y_pred, zero_division=0),
                "Acurácia": accuracy_score(y_test, y_pred),
            }
        )
        oos_records.append(pd.DataFrame({"y_proba": y_proba}, index=dates[test_idx]))

    oos_df = pd.concat(oos_records) if oos_records else pd.DataFrame(columns=["y_proba"])
    return fold_metrics, oos_df


def run_backtest(oos_df, df_features, confidence_threshold):
    """Simula uma estratégia simples a partir dos sinais out-of-sample do
    walk-forward: comprado quando o sinal indica alta, em caixa nos demais
    dias (incluindo dias fora da validação ou na "zona morta" do limiar de
    magnitude). Compara contra comprar-e-manter o ativo no mesmo período.

    Usa o retorno realizado t→t+1 da série de preços contínua (não o
    intervalo entre predições, que pode ter saltos quando há zona morta).
    """
    if oos_df.empty:
        return None

    inicio = oos_df.index.min()
    calendario = df_features.loc[df_features.index >= inicio].index

    sinal = (oos_df["y_proba"] >= confidence_threshold).astype(int)
    sinal = sinal.reindex(calendario, fill_value=0)

    retorno_futuro = df_features["Retornos"].shift(-1).reindex(calendario)

    retorno_estrategia = (sinal * retorno_futuro).fillna(0)
    retorno_buyhold = retorno_futuro.fillna(0)

    curva_estrategia = (1 + retorno_estrategia).cumprod()
    curva_buyhold = (1 + retorno_buyhold).cumprod()

    def sharpe(retornos):
        desvio = retornos.std()
        if not desvio or np.isnan(desvio):
            return None
        return (retornos.mean() / desvio) * np.sqrt(252)

    resumo = {
        "retorno_estrategia": curva_estrategia.iloc[-1] - 1,
        "retorno_buyhold": curva_buyhold.iloc[-1] - 1,
        "sharpe_estrategia": sharpe(retorno_estrategia),
        "sharpe_buyhold": sharpe(retorno_buyhold),
        "inicio": calendario.min(),
        "fim": calendario.max(),
    }
    return curva_estrategia, curva_buyhold, resumo


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

st.set_page_config(page_title="ML para Assessores de Investimento", layout="wide")

df_ticker_names = load_ticker_names()
lista_ativos = df_ticker_names["Ticker_CompanyNames"].drop_duplicates().tolist()

st.title("Aplicação de ML para Assessores de Investimento")
st.write("Ferramenta para aplicação de modelos de ML em dados de ativos da bolsa brasileira.")

with st.expander("❓ Help — Como a ferramenta funciona", expanded=False):
    tab_como, tab_vars, tab_modelos, tab_metricas = st.tabs(
        ["Como funciona", "Variáveis", "Modelos", "Métricas"]
    )

    with tab_como:
        st.markdown(
            """
Esta ferramenta tenta prever a **direção futura** de um ativo da bolsa brasileira e
transforma essa previsão em uma recomendação **Compre! / Não Compre!**.

**Passo a passo do que acontece quando você ajusta as opções na barra lateral:**

1. **Download dos dados** — baixa todo o histórico de preços do ativo (`{ticker}.SA`) no
   Yahoo Finance e filtra a partir do *ano de download* escolhido.
2. **Cálculo dos indicadores** — a partir dos preços, calcula os indicadores técnicos
   (as *variáveis* — veja a aba correspondente).
3. **Definição do alvo** — olha o retorno acumulado dos próximos *N pregões* (o
   **horizonte** escolhido). Se o *limiar de magnitude* estiver acima de 0%, dias com
   retorno futuro pequeno (dentro do limiar) são descartados como ambíguos; os demais
   são marcados como alta (`1`) ou não-alta (`0`).
4. **Validação walk-forward** — em vez de um único corte treino/teste, os dados são
   divididos em várias janelas sequenciais (**folds**): cada fold treina com todo o
   histórico disponível até aquele ponto e testa no período seguinte. O *scaler* é
   ajustado apenas no treino de cada fold, para não "espiar" o futuro. As métricas
   exibidas são a média entre os folds válidos.
5. **Backtest** — concatena as previsões de teste (fora da amostra) de todos os folds,
   em ordem temporal, e simula duas curvas de capital: seguir os sinais do modelo
   (**estratégia**) vs. simplesmente manter o ativo comprado (**comprar e manter**).
6. **Recomendação** — um modelo final é treinado com TODO o histórico rotulado
   disponível e prevê a probabilidade de alta para o dado mais recente. Se essa
   probabilidade for maior ou igual ao *limiar de confiança* escolhido, mostra
   **Compre!**; caso contrário, **Não Compre!**.

> ⚠️ **Aviso:** é uma estimativa estatística baseada apenas em indicadores técnicos e
> dados históricos. **Não é garantia de resultado** nem recomendação formal de
> investimento.
"""
        )

    with tab_vars:
        st.markdown(
            "As *variáveis* são os indicadores técnicos que alimentam o modelo. "
            "Selecione uma ou mais na barra lateral:"
        )
        for label, desc in VARIABLE_HELP.items():
            st.markdown(f"- **{label}** — {desc}")

    with tab_modelos:
        st.markdown("Modelos de machine learning disponíveis para a previsão:")
        for label, desc in MODEL_HELP.items():
            st.markdown(f"- **{label}** — {desc}")

    with tab_metricas:
        st.markdown(
            "As métricas abaixo avaliam o desempenho do modelo no **período de teste** "
            "(dados que ele não viu durante o treino):"
        )
        for label, desc in METRIC_HELP.items():
            st.markdown(f"- **{label}** — {desc}")

st.divider()

with st.sidebar:
    st.header("Configurações")

    default_stock = "BBAS3 - Banco do Brasil S.A."
    stock_complete = st.selectbox(
        "Selecione o código da ação/ativo:",
        lista_ativos,
        index=lista_ativos.index(default_stock) if default_stock in lista_ativos else 0,
    )

    year = st.selectbox(
        "Selecione desde que ano os dados serão baixados:",
        [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        index=0,  # 2016
    )

    horizon = st.selectbox(
        "Horizonte de previsão (dias):",
        [1, 5, 10, 20],
        index=0,
    )

    threshold = st.slider(
        "Limiar de magnitude do alvo (%):",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.1,
        help="Acima de 0%, dias com retorno futuro entre -limiar e +limiar são "
        "descartados do treino/teste (zona ambígua), em vez de contar como alta/queda.",
    )

    variable_labels = st.multiselect(
        "Selecione as variáveis do modelo:",
        list(VARIABLE_OPTIONS.keys()),
        default=["Relative Strength Index"],
    )
    list_variables = [VARIABLE_OPTIONS[label] for label in variable_labels]

    model_label = st.selectbox("Selecione o modelo:", list(MODEL_OPTIONS.keys()))
    model_choiced = MODEL_OPTIONS[model_label]

    st.header("Validação (walk-forward)")

    n_folds = st.slider("Número de folds:", min_value=3, max_value=10, value=5)

    confidence_threshold = st.slider(
        "Limiar de confiança do sinal:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Probabilidade mínima prevista pelo modelo para considerar 'Compre!'. "
        "Usado também nas métricas por fold e no backtest.",
    )

if len(list_variables) == 0:
    st.error("Atenção, selecione ao menos uma variável!")
    st.stop()

params = load_parameters()

# Validação dos hiperparâmetros do modelo selecionado
try:
    from sklearn.datasets import make_classification

    _x, _y = make_classification(
        n_samples=5, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False
    )
    build_model(model_choiced, params, _x, _y)
except Exception:
    st.error("Atenção, algum parâmetro inválido para o modelo selecionado!")
    st.stop()

# ---------------------------------------------------------------------------
# Pipeline de dados + modelagem
# ---------------------------------------------------------------------------

ticker = df_ticker_names[
    df_ticker_names["Ticker_CompanyNames"] == stock_complete
]["Ticker"].item()

df_prices = download_prices(ticker, year)

if df_prices.empty:
    st.error("Nenhum dado retornado para o ativo/período selecionado.")
    st.stop()

df_features = build_features(df_prices, horizon=horizon, threshold=threshold)

if "Ibov_Retorno" in list_variables:
    df_ibov = download_prices("^BVSP", year, suffix="")
    if df_ibov.empty:
        st.warning("Não foi possível obter dados do Ibovespa; a variável 'Retorno do Ibovespa' será ignorada.")
        list_variables = [v for v in list_variables if v != "Ibov_Retorno"]
    else:
        df_features["Ibov_Retorno"] = df_ibov["Adj Close"].pct_change(1).reindex(df_features.index)
        df_features.dropna(subset=["Ibov_Retorno"], inplace=True)

if len(list_variables) == 0:
    st.error("Nenhuma variável disponível após a remoção do Ibovespa (dados indisponíveis). Selecione outra variável.")
    st.stop()

if df_features.empty:
    st.error("Nenhum dado disponível após o cálculo dos indicadores para o ativo/período selecionados.")
    st.stop()

df_labeled = df_features.dropna(subset=["Alvo"])
live_row = df_features.iloc[[-1]]

if len(df_labeled) <= n_folds:
    st.error(
        "Dados insuficientes para o horizonte, limiar de magnitude e nº de folds "
        "escolhidos. Reduza um desses parâmetros ou baixe dados de um período maior."
    )
    st.stop()

fold_metrics, oos_df = run_walk_forward(
    df_labeled, list_variables, model_choiced, params, n_folds, horizon, confidence_threshold
)

if not fold_metrics:
    st.error(
        "Nenhum fold de validação pôde ser calculado com as opções atuais (dataset "
        "pequeno demais, ou algum fold ficou com uma única classe). Reduza o nº de "
        "folds, o horizonte ou o limiar de magnitude."
    )
    st.stop()

# Modelo "de produção": treinado com TODO o histórico rotulado, usado só para a
# recomendação final — nunca reaproveita scaler/modelo dos folds do walk-forward.
scaler_final = MinMaxScaler(feature_range=(0, 1))
x_all = scaler_final.fit_transform(df_labeled[list_variables].values)
y_all = df_labeled["Alvo"].values
model_final = build_model(model_choiced, params, x_all, y_all)

x_live = scaler_final.transform(live_row[list_variables].values)
proba_live = model_final.predict_proba(x_live)[0, 1]
status = "Compre!" if proba_live >= confidence_threshold else "Não Compre!"

# ---------------------------------------------------------------------------
# Saída
# ---------------------------------------------------------------------------

fig = go.Figure(data=go.Scatter(x=df_prices.index, y=df_prices["Adj Close"]))
fig.update_layout(
    title_text=f"Gráfico de Preços - {stock_complete}",
    xaxis_title="Data",
    yaxis_title="Preço [R$]",
    title_x=0.5,
)
fig.update_traces(line_color=THEME_COLOR)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Validação (walk-forward)")
df_fold_metrics = pd.DataFrame(fold_metrics)
df_fold_metrics.index = [f"Fold {i + 1}" for i in range(len(df_fold_metrics))]
media = df_fold_metrics.mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Recall", f"{media['Recall']:.2f}")
c2.metric("Precision", f"{media['Precision']:.2f}")
c3.metric("f1-score", f"{media['f1-score']:.2f}")
c4.metric("Acurácia", f"{media['Acurácia']:.2f}")
st.caption(f"Média entre {len(df_fold_metrics)} folds válidos (walk-forward, janela expansiva).")

with st.expander("Detalhe por fold"):
    st.dataframe(df_fold_metrics.round(2), use_container_width=True)

st.subheader("Backtest da estratégia")
resultado_backtest = run_backtest(oos_df, df_features, confidence_threshold)
if resultado_backtest is None:
    st.info("Sem dados suficientes de validação out-of-sample para simular o backtest.")
else:
    curva_estrategia, curva_buyhold, resumo = resultado_backtest
    st.caption(
        f"Período simulado: {resumo['inicio'].date()} a {resumo['fim'].date()} — cobre "
        "apenas o intervalo avaliado pelo walk-forward, mais curto que o histórico "
        "completo do gráfico de preços acima."
    )

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=curva_estrategia.index, y=curva_estrategia, name="Estratégia (modelo)"))
    fig_bt.add_trace(go.Scatter(x=curva_buyhold.index, y=curva_buyhold, name="Comprar e manter"))
    fig_bt.update_layout(
        title_text="Retorno acumulado — estratégia vs. comprar e manter",
        xaxis_title="Data",
        yaxis_title="Capital acumulado (base 1,0)",
        title_x=0.5,
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    def _fmt_sharpe(v):
        return "N/A" if v is None else f"{v:.2f}"

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Retorno estratégia", f"{resumo['retorno_estrategia'] * 100:.1f}%")
    b2.metric("Retorno comprar e manter", f"{resumo['retorno_buyhold'] * 100:.1f}%")
    b3.metric("Sharpe estratégia*", _fmt_sharpe(resumo["sharpe_estrategia"]))
    b4.metric("Sharpe comprar e manter*", _fmt_sharpe(resumo["sharpe_buyhold"]))
    st.caption("*Sharpe simplificado (sem taxa livre de risco), anualizado por √252.")

st.subheader("Recomendação")
recomendacao_texto = f"**{status}** (probabilidade de alta: {proba_live:.0%}, limiar: {confidence_threshold:.0%})"
if status == "Compre!":
    st.success(recomendacao_texto)
else:
    st.error(recomendacao_texto)
