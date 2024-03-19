import random
from pandas_datareader import data as pdr
import yfinance as yf
import matplotlib.pyplot as plt

# data = yf.download("AAPL", start="2020-01-01", end="2021-01-01")
# print(data.head())

apple = yf.Ticker("SGHT")
print(apple.info)
# data['Close'].plot()
# plt.show()


# we can have ticker, open & close for time period
# industry, industryKey, sector, sectorKey, 
# overallrisk, boardRisk, compensationRisk, shareHolderRightsRisk. Ranked 1-10
