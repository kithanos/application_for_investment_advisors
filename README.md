# PA-ML_Appication_for_Investment_Advisors

Códigos relacionados ao projeto aplicado — *"Ferramenta para aplicação de modelos de Machine Learning para Assessores de Investimento"* — do aluno Lucas Kitano, desenvolvido para fins de conclusão do curso de MBA de Ciência de Dados para o Mercado Financeiro.

## Sobre

Ferramenta web interativa que aplica modelos de Machine Learning sobre dados de ativos da bolsa brasileira (B3/IBOV) para estimar se o ativo tende a subir no dia seguinte, gerando uma recomendação de "Compre!" / "Não Compre!".

O usuário seleciona o ativo, o período dos dados, as variáveis (indicadores técnicos) e o modelo. A aplicação baixa o histórico de preços, calcula os indicadores, treina o modelo e exibe o gráfico de preços junto com as métricas de desempenho.

- **Interface:** [Streamlit](https://streamlit.io/) (`streamlit_app.py`)
- **Dados de mercado:** [`yfinance`](https://github.com/ranaroussi/yfinance)
- **Indicadores técnicos:** [`TA-Lib`](https://github.com/TA-Lib/ta-lib-python) (RSI, médias móveis, ADX, padrões de candlestick, etc.)
- **Machine Learning:** [`scikit-learn`](https://scikit-learn.org/) — Regressão Logística, Árvore de Decisão, Random Forest e Rede Neural (MLP)

Os hiperparâmetros de cada modelo são configurados no arquivo `parameters.json`.

## Como executar

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

> No Windows, o `TA-Lib` normalmente requer o binário C ou o wheel pré-compilado instalado previamente.

## Estrutura

```
streamlit_app.py    # aplicação Streamlit
parameters.json     # hiperparâmetros dos modelos
requirements.txt    # dependências
dataset/            # códigos (tickers) e nomes dos ativos do IBOV
```
