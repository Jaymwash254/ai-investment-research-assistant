import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.covariance import LedoitWolf
import random

# --- UI Setup ---
st.set_page_config(page_title="AI Investment Research Assistant", layout="wide", initial_sidebar_state="expanded")

st.sidebar.title("📈 AI Investment Research Assistant")
page = st.sidebar.radio("Navigate", ["Market Sentiment", "Portfolio Optimizer", "Stock Forecast", "Financial Chatbot"])

# --- Dummy Data ---
stocks_list = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"]
sentiments = {"AAPL": "Positive", "TSLA": "Neutral", "GOOGL": "Negative", "MSFT": "Positive", "AMZN": "Neutral"}

@st.cache_data
def load_stock_data(stock):
    dates = pd.date_range(end=pd.Timestamp.today(), periods=90)
    prices = np.cumsum(np.random.randn(90)) + 100
    return pd.DataFrame({"ds": dates, "y": prices})

# --- Pages ---
if page == "Market Sentiment":
    st.title("📰 Market Sentiment Analysis")
    selected_stock = st.selectbox("Select a Stock", stocks_list)
    st.metric(label="Sentiment", value=sentiments.get(selected_stock, "Neutral"))

elif page == "Portfolio Optimizer":
    st.title("🛡️ Portfolio Optimizer")
    selected_stocks = st.multiselect("Select Stocks to Include", stocks_list)
    risk_appetite = st.radio("Select Risk Appetite", ["Low", "Medium", "High"])
    if st.button("Optimize Portfolio"):
        if selected_stocks:
            weights = np.random.dirichlet(np.ones(len(selected_stocks)))
            portfolio = pd.DataFrame({"Stock": selected_stocks, "Allocation %": weights * 100})
            st.dataframe(portfolio)
            st.write("📊 Allocation Pie Chart:")
            st.plotly_chart(portfolio.set_index("Stock").plot.pie(y="Allocation %", autopct='%1.1f%%', legend=False, figsize=(5,5)).figure)
        else:
            st.error("Please select at least one stock.")

elif page == "Stock Forecast":
    st.title("🔮 Stock Price Forecast")
    selected_stock = st.selectbox("Select a Stock for Forecast", stocks_list, key="forecast")
    stock_data = load_stock_data(selected_stock)
    st.line_chart(stock_data.set_index('ds')['y'], use_container_width=True)
    
    if st.button("Run Forecast"):
        model = Prophet()
        model.fit(stock_data)
        future = model.make_future_dataframe(periods=7)
        forecast = model.predict(future)
        
        st.write("📈 Forecasted Prices:")
        fig = model.plot(forecast)
        st.pyplot(fig)

elif page == "Financial Chatbot":
    st.title("💬 Financial Chatbot")
    user_input = st.text_input("Ask me something (e.g., Top 3 stocks?)")
    if user_input:
        if "top" in user_input.lower():
            st.success("🔥 Top Stocks Today: AAPL, MSFT, NVDA")
        elif "sentiment" in user_input.lower():
            st.info("📈 General market sentiment is Positive.")
        else:
            st.warning("🤔 I don't understand. Try asking about 'top stocks' or 'market sentiment'.")
