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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

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
}

MODEL_OPTIONS = {
    "Logistic Regression": "lr",
    "Decision Tree": "dt",
    "Random Forest": "rf",
    "Rede Neural": "mlp",
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
}

METRIC_HELP = {
    "Recall": "Dos pregões que realmente subiram, qual fração o modelo acertou. Recall alto = "
    "perde poucas altas.",
    "Precision": "Das vezes que o modelo disse 'Compre!', qual fração de fato subiu. Precision "
    "alta = poucos alarmes falsos.",
    "f1-score": "Média harmônica entre Precision e Recall — equilíbrio entre os dois.",
    "Acurácia": "Proporção total de previsões corretas (altas e quedas) sobre o conjunto de teste.",
}


@st.cache_data
def load_ticker_names():
    """Lê o dataset com o código (ticker) e o nome do ativo."""
    df = pd.read_csv(DATASET_FILENAME, sep=";")
    df["Ticker_CompanyNames"] = df["Ticker"] + " - " + df["Company_Name"]
    return df


@st.cache_data
def load_parameters():
    """Carrega os hiperparâmetros dos modelos do parameters.json."""
    with open(PARAMETERS_FILENAME) as f:
        return json.load(f)


@st.cache_data(show_spinner="Baixando dados do ativo...")
def download_prices(ticker, year):
    """Baixa o histórico de preços do ativo via Yahoo Finance."""
    # period="max": versões recentes do yfinance passaram a retornar só ~1 mês
    # quando período/datas não são informados, o que zerava os indicadores longos.
    df = yf.download(f"{ticker}.SA", period="max", auto_adjust=False, progress=False)

    # yfinance pode retornar colunas MultiIndex para um único ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Filtra os dados de acordo com o ano selecionado pelo usuário
    df = df.loc[df.index > f"{year}-12-31"]
    return df


def build_features(df_prices):
    """Calcula os indicadores técnicos e a variável alvo."""
    df = df_prices.copy()
    df["Retornos"] = df["Adj Close"].pct_change(1)
    df.dropna(axis=0, inplace=True)
    df["Alvo"] = np.where(df["Retornos"].shift(-1) > 0, 1, 0)

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

    df.dropna(axis=0, inplace=True)
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
    else:
        raise ValueError(f"Modelo desconhecido: {model_choiced}")

    return model.fit(x_train, y_train)


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
Esta ferramenta tenta prever a **direção do próximo pregão** de um ativo da bolsa brasileira
(vai subir ou não) e transforma essa previsão em uma recomendação **Compre! / Não Compre!**.

**Passo a passo do que acontece quando você ajusta as opções na barra lateral:**

1. **Download dos dados** — baixa todo o histórico de preços do ativo (`{ticker}.SA`) no
   Yahoo Finance e filtra a partir do *ano de download* escolhido.
2. **Cálculo dos indicadores** — a partir dos preços, calcula os indicadores técnicos
   (as *variáveis* — veja a aba correspondente).
3. **Definição do alvo** — para cada dia, o alvo é `1` se o retorno do **dia seguinte** for
   positivo, e `0` caso contrário.
4. **Treino e teste** — os dados são escalados e divididos por data: tudo até o *ano de treino*
   é usado para **treinar** o modelo; o período posterior é usado para **testar**. O scaler é
   ajustado apenas no treino, para não "espiar" o futuro.
5. **Recomendação** — o modelo faz a previsão para a amostra mais recente. Se prevê alta,
   mostra **Compre!**; caso contrário, **Não Compre!**.

> ⚠️ **Aviso:** é uma estimativa estatística de curtíssimo prazo, baseada apenas em indicadores
> técnicos. **Não é garantia de resultado** nem recomendação formal de investimento.
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

    end_train = st.selectbox(
        "Selecione até que ano os dados serão treinados:",
        [2020, 2021, 2022, 2023, 2024, 2025],
        index=2,  # 2022
    )

    variable_labels = st.multiselect(
        "Selecione as variáveis do modelo:",
        list(VARIABLE_OPTIONS.keys()),
        default=["Relative Strength Index"],
    )
    list_variables = [VARIABLE_OPTIONS[label] for label in variable_labels]

    model_label = st.selectbox("Selecione o modelo:", list(MODEL_OPTIONS.keys()))
    model_choiced = MODEL_OPTIONS[model_label]

# Validações (equivalentes aos ConfirmDialog do Dash)
if end_train <= year:
    st.warning("Atenção, escolha uma data de treinamento superior a data de download dos dados!")

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

df_features = build_features(df_prices)

selected = list_variables + ["Alvo"]
df_selected = df_features[selected].copy()

if end_train <= year:
    end_train = year + 1

# Divide treino/teste ANTES de escalar, para evitar vazamento de dados
# (o scaler é ajustado apenas no treino e depois aplicado ao teste).
df_train_raw = df_selected.loc[df_selected.index <= f"{end_train}-01-01"]
df_test_raw = df_selected.loc[df_selected.index > f"{end_train}-01-01"]

if df_test_raw.empty:
    st.error(
        "Não há dados de teste para o corte escolhido. "
        "Selecione um ano de treino anterior ao último ano disponível."
    )
    st.stop()

x_train_raw, y_train = df_train_raw.drop("Alvo", axis=1), df_train_raw["Alvo"]
x_test_raw, y_test = df_test_raw.drop("Alvo", axis=1), df_test_raw["Alvo"]

scaler = MinMaxScaler(feature_range=(0, 1))
x_train = pd.DataFrame(
    scaler.fit_transform(x_train_raw.values),
    index=x_train_raw.index,
    columns=x_train_raw.columns,
)
x_test = pd.DataFrame(
    scaler.transform(x_test_raw.values),
    index=x_test_raw.index,
    columns=x_test_raw.columns,
)

model = build_model(model_choiced, params, x_train, y_train)
y_pred = model.predict(x_test)

recall = np.round(recall_score(y_test, y_pred), 2)
precision = np.round(precision_score(y_test, y_pred), 2)
f1 = np.round(f1_score(y_test, y_pred), 2)
acc = np.round(accuracy_score(y_test, y_pred), 2)
status = "Compre!" if y_pred[-1] > 0 else "Não Compre!"

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

st.subheader("Métricas de desempenho")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Recall", recall)
c2.metric("Precision", precision)
c3.metric("f1-score", f1)
c4.metric("Acuraccy", acc)

st.subheader("Recomendação")
if status == "Compre!":
    st.success(f"**{status}**")
else:
    st.error(f"**{status}**")
