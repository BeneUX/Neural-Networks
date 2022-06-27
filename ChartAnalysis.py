import pandas as pd
import numpy as np
import yfinance as yf

# Import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#%matplotlib inline

# Fetch the stock data
company = 'META'
data = yf.download(company, '2015-01-01')

# Visualise the data
data['Close'].plot(figsize=(15,7))
plt.ylabel('Close Price')
plt.title(company + ' Close Price')
plt.show()

# Calculate exponential moving average
data['12d_EMA'] = data.Close.ewm(span=12, adjust=False).mean()
data['26d_EMA'] = data.Close.ewm(span=26, adjust=False).mean()

# Plot Close Price and EMA
data[['Close', '12d_EMA', '26d_EMA']][-100:].plot(figsize=(15, 7))
plt.ylabel('Price')
plt.title('EMA of Close Price')
plt.show()

# Calculate MACD line
data['macd'] = data['12d_EMA']- data['26d_EMA']

# Calculate MACD Signal line
data['macdsignal'] = data.macd.ewm(span=9, adjust=False).mean()

# Plot MACD and MACD Signal line
data[['macd','macdsignal']][-100:].plot(figsize=(15,7))
plt.ylabel('Price')
plt.title('MACD and MACD Signal Line')
plt.show()

# Column to store trading signals
data['trading_signal'] = np.nan

# Buy signals
data.loc[data['macd'] > data['macdsignal'], 'trading_signal'] = 1

# Sell signals
data.loc[data['macd'] < data['macdsignal'], 'trading_signal'] = -1

# Fill the missing values with last valid observation
data = data.fillna(method = 'ffill')

data.tail()

# ----------------------------------------------------------------
# --------------- Kennzahlen berechnen und ausgeben --------------
# ----------------------------------------------------------------

# Calculate daily returns of Apple
data['returns'] = data.Close.pct_change()

# Calculate daily strategy returns
data['strategy_returns'] = data.returns * data.trading_signal.shift(1)

# Calculate cumulative strategy returns
cumulative_strategy_returns = (data.strategy_returns + 1).cumprod()

# Plot cumulative strategy returns
cumulative_strategy_returns.plot(figsize=(15,7))
plt.ylabel('Cumulative Returns')
plt.title('Cumulative Returns of MACD Strategy on ' + company)
plt.show()


# Total number of trading days
days = len(cumulative_strategy_returns)

# Calculate compounded annual growth rate
annual_returns = (cumulative_strategy_returns.iloc[-1]**(252/days) - 1)*100

print('The CAGR is %.2f%%' % annual_returns)

# Calculate the annualised volatility
annual_volatility = data.strategy_returns.std() * np.sqrt(252) * 100

print('The annualised volatility is %.2f%%' % annual_volatility)

# Assume the annual risk-free rate is 2%
risk_free_rate = 0.02
daily_risk_free_return = risk_free_rate/252

# Calculate the excess returns by subtracting the daily returns by daily risk-free return
excess_daily_returns = data.strategy_returns - daily_risk_free_return

# Calculate the sharpe ratio using the given formula
sharpe_ratio = (excess_daily_returns.mean() /
                excess_daily_returns.std()) * np.sqrt(252)

print('The Sharpe ratio is %.2f' % sharpe_ratio)