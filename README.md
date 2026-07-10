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

## Pré-requisitos

- **Python 3.12** (ou superior)
- Conexão com a **internet** — o app baixa os preços dos ativos via Yahoo Finance a cada execução.

## Instalação (primeira vez)

Recomenda-se usar um ambiente virtual. No **PowerShell** (Windows), a partir da pasta do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

No Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **TA-Lib no Windows:** versões recentes já possuem *wheel* pré-compilado no PyPI, então o `pip install` normalmente funciona sem instalar o binário C manualmente.

## Como executar

Com o ambiente virtual ativado:

```powershell
streamlit run streamlit_app.py
```

O navegador abre automaticamente em **http://localhost:8501**. Para parar o servidor, pressione `Ctrl+C` no terminal.

Sem ativar o ambiente, é possível rodar em uma linha só (Windows):

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

> Se o Windows bloquear o `Activate.ps1` por política de execução, rode uma vez:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` (ou use a forma de uma linha só acima).

## Estrutura

```
streamlit_app.py    # aplicação Streamlit
parameters.json     # hiperparâmetros dos modelos
requirements.txt    # dependências
dataset/            # códigos (tickers) e nomes dos ativos do IBOV
```
