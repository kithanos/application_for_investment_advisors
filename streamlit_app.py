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

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df_selected.drop("Alvo", axis=1).values)
df_scaled = pd.DataFrame(
    scaled, index=df_selected.index, columns=df_selected.drop("Alvo", axis=1).columns
)
df_scaled["Alvo"] = df_selected["Alvo"]

if end_train <= year:
    end_train = year + 1

df_train = df_scaled.loc[df_scaled.index <= f"{end_train}-01-01"]
df_test = df_scaled.loc[df_scaled.index > f"{end_train}-01-01"]

x_train = df_train.drop(["Alvo"], axis=1)
y_train = df_train["Alvo"]
x_test = df_test.drop(["Alvo"], axis=1)
y_test = df_test["Alvo"]

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
