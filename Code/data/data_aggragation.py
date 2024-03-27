import random
from pandas_datareader import data as pdr
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

    
# required_data = []
# with open(r'Code\data\needed_catagories.txt', 'r') as f:
#     for line in f:
#         required_data.append(line.strip())

# print(required_data)


tickrs = []
catagories = ['Ticker', 'timeseries', 'auditRisk', 'boardRisk', 'compensationRisk', 'shareHolderRightsRisk', 'overallRisk', 'governanceEpochDate', 'compensationAsOfEpochDate', 'maxAge', 'previousClose', 'open', 'dayLow', 'dayHigh', 'regularMarketPreviousClose', 'regularMarketOpen', 'regularMarketDayLow', 'regularMarketDayHigh', 'dividendRate', 'dividendYield', 'exDividendDate', 'payoutRatio', 'fiveYearAvgDividendYield', 'beta', 'trailingPE', 'forwardPE', 'volume', 'regularMarketVolume', 'averageVolume', 'averageVolume10days', 'averageDailyVolume10Day', 'marketCap', 'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'priceToSalesTrailing12Months', 'fiftyDayAverage', 'twoHundredDayAverage', 'trailingAnnualDividendRate', 'trailingAnnualDividendYield', 'enterpriseValue', 'profitMargins', 'floatShares', 'sharesOutstanding', 'sharesShort', 'sharesShortPriorMonth', 'sharesShortPreviousMonthDate', 'dateShortInterest', 'sharesPercentSharesOut', 'heldPercentInsiders', 'heldPercentInstitutions', 'shortRatio', 'shortPercentOfFloat', 'impliedSharesOutstanding', 'bookValue', 'priceToBook', 'lastFiscalYearEnd', 'nextFiscalYearEnd', 'mostRecentQuarter', 'earningsQuarterlyGrowth', 'netIncomeToCommon', 'trailingEps', 'forwardEps:', 'pegRatio', 'lastSplitFactor', 'lastSplitDate', 'enterpriseToRevenue', 'enterpriseToEbitda', '52WeekChange', 'SandP52WeekChange', 'lastDividendValue', 'lastDividendDate', 'firstTradeDateEpochUtc', 'currentPrice', 'targetHighPrice', 'targetLowPrice', 'targetMeanPrice', 'targetMedianPrice', 'recommendationMean', 'recommendationKey', 'numberOfAnalystOpinions', 'totalCash', 'totalCashPerShare', 'ebitda', 'totalDebt', 'quickRatio', 'currentRatio', 'totalRevenue', 'debtToEquity', 'revenuePerShare', 'returnOnAssets', 'returnOnEquity', 'freeCashflow', 'operatingCashflow', 'earningsGrowth', 'revenueGrowth', 'grossMargins', 'ebitdaMargins', 'operatingMargins', 'trailingPegRatio']
df = pd.DataFrame(columns=catagories)

for tickr_name in tickrs:
    tickr = yf.Ticker(tickr_name)
    data_to_append = [tickr_name]
    for catagory in catagories:
        if catagory == 'timeseries':
            data = yf.download(tickr_name, start="2020-01-01", end="2021-01-01")
        else:
            data = tickr[catagory]
        data_to_append.append(data)
        
    df = df.append(pd.Series(data_to_append, index=df.columns), ignore_index=True)
        
        

# data = yf.download("AAPL", start="2020-01-01", end="2021-01-01")
# print(data.head())

apple = yf.Ticker("SGHT")
print(apple.info[catagories[0]])
# data['Close'].plot()
# plt.show()


# we can have ticker, open & close for time period
# industry, industryKey, sector, sectorKey, 
# overallrisk, boardRisk, compensationRisk, shareHolderRightsRisk. Ranked 1-10

# Data Josh wants:

