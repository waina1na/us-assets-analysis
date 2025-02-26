import streamlit as st
from alpha_vantage.timeseries import TimeSeries
from pycoingecko import CoinGeckoAPI
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Title of the app
st.title("💰 Asset Class Analysis: Wealth Building Strategies")

# Sidebar for user inputs
st.sidebar.header("🎛️ User Inputs")
start_date = st.sidebar.date_input("📅 Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("📅 End Date", pd.to_datetime("2023-10-01"))
initial_investment = st.sidebar.number_input("💵 Initial Investment ($)", value=10000, min_value=1000, step=1000)
num_simulations = st.sidebar.slider("🔢 Number of Monte Carlo Simulations", min_value=100, max_value=5000, value=1000)
time_horizon = st.sidebar.slider("⏳ Time Horizon (Years)", min_value=1, max_value=30, value=10)
api_key = st.sidebar.text_input("🔑 Alpha Vantage API Key", type="password")

# Define asset classes and their tickers
tickers = {
    "Equities": "SPY",  # S&P 500 ETF
    "Fixed Income": "TLT",  # Long-term Treasury Bond ETF
    "Commodities": "GLD",  # Gold ETF
    "Real Estate": "VNQ",  # Real Estate ETF
    "Cash": "SHY",  # Short-term Treasury Bond ETF (proxy for cash)
}

# Fetch historical data from Alpha Vantage
@st.cache_data  # Cache data to improve performance
def fetch_alpha_vantage_data(tickers, start_date, end_date, api_key):
    ts = TimeSeries(key=api_key, output_format="pandas")
    data = pd.DataFrame()

    for asset, ticker in tickers.items():
        try:
            df, _ = ts.get_daily_adjusted(symbol=ticker, outputsize="full")
            df = df.loc[start_date:end_date]
            df = df.rename(columns={"5. adjusted close": asset})
            data[asset] = df[asset]
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()  # Return an empty DataFrame to avoid further errors

    return data

# Fetch Bitcoin data from CoinGecko
@st.cache_data  # Cache data to improve performance
def fetch_bitcoin_data(start_date, end_date):
    cg = CoinGeckoAPI()
    bitcoin_data = cg.get_coin_market_chart_by_id(id="bitcoin", vs_currency="usd", days="max")
    btc_prices = pd.DataFrame(bitcoin_data["prices"], columns=["timestamp", "price"])
    btc_prices["timestamp"] = pd.to_datetime(btc_prices["timestamp"], unit="ms")
    btc_prices.set_index("timestamp", inplace=True)
    btc_prices = btc_prices.loc[start_date:end_date]
    return btc_prices

# Main program
def main():
    if not api_key:
        st.error("Please enter your Alpha Vantage API key in the sidebar.")
        return

    # Fetch data
    st.write("📊 Fetching data...")
    alpha_vantage_data = fetch_alpha_vantage_data(tickers, start_date, end_date, api_key)
    if alpha_vantage_data.empty:
        st.error("Failed to fetch Alpha Vantage data. Please check the tickers and date range.")
        return  # Stop execution if data fetching fails

    bitcoin_data = fetch_bitcoin_data(start_date, end_date)

    # Combine data
    combined_data = alpha_vantage_data.copy()
    combined_data["Bitcoin"] = bitcoin_data["price"].resample("D").ffill()  # Resample Bitcoin data to daily

    # Calculate daily returns
    returns = combined_data.pct_change().dropna()

    # Display correlation matrix
    st.write("### 📈 Correlation Matrix")
    st.write("This table shows how different asset classes move in relation to each other.")
    correlation_matrix = returns.corr()
    st.dataframe(correlation_matrix.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1))

    # Visualize correlation matrix
    st.write("### 🌈 Correlation Heatmap")
    st.write("This heatmap provides a visual representation of the correlation matrix.")
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    st.pyplot(plt)

    # Monte Carlo Simulation
    st.write("### 🎲 Monte Carlo Simulation")
    st.write(f"This simulation shows potential portfolio values over {time_horizon} years based on historical returns.")
    portfolio_values = monte_carlo_simulation(returns, initial_investment, num_simulations, time_horizon)

    # Plot Monte Carlo results
    plt.figure(figsize=(10, 6))
    for i in range(num_simulations):
        plt.plot(portfolio_values[i], color="blue", alpha=0.1)
    plt.title(f"Monte Carlo Simulation of Portfolio Value Over {time_horizon} Years")
    plt.xlabel("Years")
    plt.ylabel("Portfolio Value ($)")
    st.pyplot(plt)

    # Markov Chain Analysis
    st.write("### 🔄 Markov Chain Analysis")
    st.write("This analysis shows the probability of transitioning between positive and negative return states.")
    transition_matrix = markov_chain_analysis(returns)
    st.write("#### Transition Matrix (Positive vs. Negative Returns)")
    st.dataframe(transition_matrix.style.background_gradient(cmap="Blues"))

    # Interpret Markov Chain results
    st.write("#### 📝 Interpretation:")
    st.write("- **Positive to Positive**: Probability of staying in a positive return state.")
    st.write("- **Positive to Negative**: Probability of transitioning from positive to negative returns.")
    st.write("- **Negative to Positive**: Probability of transitioning from negative to positive returns.")
    st.write("- **Negative to Negative**: Probability of staying in a negative return state.")

# Run the app
if __name__ == "__main__":
    main()
