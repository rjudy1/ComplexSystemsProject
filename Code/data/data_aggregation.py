import pandas as pd
import yfinance as yf
import csv
import numpy as np
import time

tickrs = []

with open('Code\\data\\valid_tickers.csv', 'r') as f:
    all_tickers = list(csv.reader(f))
    random_indices = np.random.choice(len(all_tickers), 1000, replace=False)

    for i in random_indices:
        tickrs.append(all_tickers[i][0])


catagories = ['dates','Ticker','auditRisk', 'boardRisk', 'compensationRisk', 'shareHolderRightsRisk', 'overallRisk', 'governanceEpochDate', 'compensationAsOfEpochDate', 'maxAge', 'previousClose', 'open', 'dayLow', 'dayHigh', 'regularMarketPreviousClose', 'regularMarketOpen', 'regularMarketDayLow', 'regularMarketDayHigh', 'dividendRate', 'dividendYield', 'exDividendDate', 'payoutRatio', 'fiveYearAvgDividendYield', 'beta', 'trailingPE', 'forwardPE', 'volume', 'regularMarketVolume', 'averageVolume', 'averageVolume10days', 'averageDailyVolume10Day', 'marketCap', 'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'priceToSalesTrailing12Months', 'fiftyDayAverage', 'twoHundredDayAverage', 'trailingAnnualDividendRate', 'trailingAnnualDividendYield', 'enterpriseValue', 'profitMargins', 'floatShares', 'sharesOutstanding', 'sharesShort', 'sharesShortPriorMonth', 'sharesShortPreviousMonthDate', 'dateShortInterest', 'sharesPercentSharesOut', 'heldPercentInsiders', 'heldPercentInstitutions', 'shortRatio', 'shortPercentOfFloat', 'impliedSharesOutstanding', 'bookValue', 'priceToBook', 'lastFiscalYearEnd', 'nextFiscalYearEnd', 'mostRecentQuarter', 'earningsQuarterlyGrowth', 'netIncomeToCommon', 'trailingEps', 'forwardEps', 'pegRatio', 'lastSplitFactor', 'lastSplitDate', 'enterpriseToRevenue', 'enterpriseToEbitda', '52WeekChange', 'SandP52WeekChange', 'lastDividendValue', 'lastDividendDate', 'firstTradeDateEpochUtc', 'currentPrice', 'targetHighPrice', 'targetLowPrice', 'targetMeanPrice', 'targetMedianPrice', 'recommendationMean', 'recommendationKey', 'numberOfAnalystOpinions', 'totalCash', 'totalCashPerShare', 'ebitda', 'totalDebt', 'quickRatio', 'currentRatio', 'totalRevenue', 'debtToEquity', 'revenuePerShare', 'returnOnAssets', 'returnOnEquity', 'freeCashflow', 'operatingCashflow', 'earningsGrowth', 'revenueGrowth', 'grossMargins', 'ebitdaMargins', 'operatingMargins', 'trailingPegRatio']
df = pd.DataFrame(columns=catagories)

big_data = []
tickers_row = []

for tickr_name in tickrs:
    tickr = yf.Ticker(tickr_name)
    data_to_append = []
    for catagory in catagories:
        if catagory == 'dates':
            try:
                data_large = yf.download(tickr_name, start="1970-01-01", end="2020-01-01")['Adj Close']
                dates = list(data_large.index.strftime('%m/%d/%Y'))
                values = data_large.values
                data_to_append.append(dates)
                data = list(values)
            except:
                break
        elif catagory == 'Ticker':
            pass
        else:
            try:
                data = tickr.info[catagory]
            except:
                data = None
        
        if catagory == 'Ticker':
            data_to_append.insert(0, tickr_name)
        else:  
            data_to_append.append(data)
            
    if data_to_append:
        big_data.append(data_to_append)
        
    time.sleep(1)

catagories[1], catagories[0] = catagories[0], catagories[1]
catagories.insert(2, 'time_series')
df = pd.DataFrame(big_data, columns=catagories)

df = f.set_index('Ticker', inplace=True)

df.to_csv('Code\\data\\ticker_info.csv')     


