import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

st.set_page_config(layout="wide")

st.title("MVP - Otimização e Screening de Carteiras")

# Funções auxiliares
def get_data(tickers, period="1y"):
    data = yf.download(tickers, period=period)["Adj Close"]
    return data.dropna()

def calculate_metrics(data):
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    return returns, mean_returns, cov_matrix

def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, volatility

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    p_returns, p_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_returns - risk_free_rate) / p_volatility

def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.0):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(negative_sharpe_ratio, num_assets * [1. / num_assets], args=args, 
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def generate_portfolios(mean_returns, cov_matrix, num_portfolios=5000):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        returns, volatility = portfolio_performance(weights, mean_returns, cov_matrix)
        sharpe_ratio = (returns) / volatility
        results[0,i] = returns
        results[1,i] = volatility
        results[2,i] = sharpe_ratio
        weights_record.append(weights)
    return results, weights_record

# Aba: Fronteira Eficiente com Heatmap
def efficient_frontier_tab():
    st.header("Fronteira Eficiente e Heatmap de Covariância")
    tickers = st.text_input("Digite os tickers separados por espaço (ex: PETR4.SA VALE3.SA AAPL):", "PETR4.SA VALE3.SA")
    period = st.selectbox("Selecione o período:", ["1y", "2y", "5y", "10y"])

    if tickers:
        tickers_list = tickers.upper().split()
        data = get_data(tickers_list, period=period)
        returns, mean_returns, cov_matrix = calculate_metrics(data)

        st.subheader("Heatmap de Covariância")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(cov_matrix.round(3), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.subheader("Fronteira Eficiente")
        results, _ = generate_portfolios(mean_returns, cov_matrix)
        max_sharpe_idx = np.argmax(results[2])

        fig, ax = plt.subplots(figsize=(10,6))
        scatter = ax.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', marker='o')
        ax.scatter(results[1,max_sharpe_idx], results[0,max_sharpe_idx], marker='*', color='r', s=500, label='Melhor Sharpe')
        ax.set_xlabel('Volatilidade')
        ax.set_ylabel('Retorno Esperado')
        ax.set_title('Fronteira Eficiente')
        fig.colorbar(scatter, label='Índice de Sharpe')
        ax.legend()
        st.pyplot(fig)

# Aba: Screening de Ações
def screening_tab():
    st.header("Screening de Ações")
    tickers = st.text_input("Digite os tickers separados por espaço:", "PETR4.SA VALE3.SA")
    period = st.selectbox("Período de avaliação:", ["1y", "2y", "5y"])

    retorno_min = st.number_input("Retorno mínimo (%)", value=0.0)
    volatilidade_max = st.number_input("Volatilidade máxima (%)", value=50.0)

    if st.button("Executar Screening") and tickers:
        tickers_list = tickers.upper().split()
        data = get_data(tickers_list, period=period)
        _, mean_returns, cov_matrix = calculate_metrics(data)

        filtered = [(ticker, round(mean_returns[ticker]*100,3), round(np.sqrt(cov_matrix.loc[ticker,ticker])*100,3)) 
                    for ticker in tickers_list 
                    if mean_returns[ticker]*100 >= retorno_min and np.sqrt(cov_matrix.loc[ticker,ticker])*100 <= volatilidade_max]

        df = pd.DataFrame(filtered, columns=["Ticker", "Retorno (%)", "Volatilidade (%)"])
        st.dataframe(df)

# Aba: Backtest
def backtest_tab():
    st.header("Backtest de Carteira")
    tickers = st.text_input("Digite os tickers separados por espaço:", "PETR4.SA VALE3.SA")
    period = st.selectbox("Selecione o período de backtest:", ["1y", "2y", "5y", "10y"])

    if st.button("Executar Backtest") and tickers:
        tickers_list = tickers.upper().split()
        data = get_data(tickers_list, period=period)
        normalized = data / data.iloc[0]

        st.subheader("Evolução do Investimento")
        fig, ax = plt.subplots(figsize=(10,6))
        for ticker in tickers_list:
            ax.plot(normalized.index, normalized[ticker], label=ticker)
        ax.set_ylabel("Retorno Acumulado")
        ax.set_xlabel("Data")
        ax.legend()
        st.pyplot(fig)

# Layout com abas
tabs = st.tabs(["Fronteira Eficiente", "Screening", "Backtest"])

with tabs[0]:
    efficient_frontier_tab()
with tabs[1]:
    screening_tab()
with tabs[2]:
    backtest_tab()
