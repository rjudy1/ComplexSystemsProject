import pandas as pd
import yfinance as yf
import csv
import numpy as np
import time

tickrs = []

with open('data\\valid_tickers.csv', 'r') as f:
    all_tickers = list(csv.reader(f))
    random_indices = np.random.permutation(len(all_tickers))

    for i in random_indices:
        tickrs.append(all_tickers[i][0])


catagories = ['dates','Ticker','auditRisk', 'boardRisk', 'compensationRisk', 'shareHolderRightsRisk', 'overallRisk', 'governanceEpochDate', 'compensationAsOfEpochDate', 'maxAge', 'previousClose', 'open', 'dayLow', 'dayHigh', 'regularMarketPreviousClose', 'regularMarketOpen', 'regularMarketDayLow', 'regularMarketDayHigh', 'dividendRate', 'dividendYield', 'exDividendDate', 'payoutRatio', 'fiveYearAvgDividendYield', 'beta', 'trailingPE', 'forwardPE', 'volume', 'regularMarketVolume', 'averageVolume', 'averageVolume10days', 'averageDailyVolume10Day', 'marketCap', 'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'priceToSalesTrailing12Months', 'fiftyDayAverage', 'twoHundredDayAverage', 'trailingAnnualDividendRate', 'trailingAnnualDividendYield', 'enterpriseValue', 'profitMargins', 'floatShares', 'sharesOutstanding', 'sharesShort', 'sharesShortPriorMonth', 'sharesShortPreviousMonthDate', 'dateShortInterest', 'sharesPercentSharesOut', 'heldPercentInsiders', 'heldPercentInstitutions', 'shortRatio', 'shortPercentOfFloat', 'impliedSharesOutstanding', 'bookValue', 'priceToBook', 'lastFiscalYearEnd', 'nextFiscalYearEnd', 'mostRecentQuarter', 'earningsQuarterlyGrowth', 'netIncomeToCommon', 'trailingEps', 'forwardEps', 'pegRatio', 'lastSplitFactor', 'lastSplitDate', 'enterpriseToRevenue', 'enterpriseToEbitda', '52WeekChange', 'SandP52WeekChange', 'lastDividendValue', 'lastDividendDate', 'firstTradeDateEpochUtc', 'currentPrice', 'targetHighPrice', 'targetLowPrice', 'targetMeanPrice', 'targetMedianPrice', 'recommendationMean', 'recommendationKey', 'numberOfAnalystOpinions', 'totalCash', 'totalCashPerShare', 'ebitda', 'totalDebt', 'quickRatio', 'currentRatio', 'totalRevenue', 'debtToEquity', 'revenuePerShare', 'returnOnAssets', 'returnOnEquity', 'freeCashflow', 'operatingCashflow', 'earningsGrowth', 'revenueGrowth', 'grossMargins', 'ebitdaMargins', 'operatingMargins', 'trailingPegRatio']
df = pd.DataFrame(columns=catagories)

big_data = []
tickers_row = []
i = -1

while len(big_data) < 400:
    if i > len(tickrs) - 1:
        break
    
    i += 1
    tickr = yf.Ticker(tickrs[i])
    data_to_append = []
    do_append = True
    for catagory in catagories:
        if catagory == 'dates':
            try:
                data_large = yf.download(tickrs[i], start="1970-01-01", end="2022-01-01")['Adj Close']
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
                if catagory == 'forwardEps' or catagory == 'dividendRate' or catagory == 'earningsGrowth':
                    do_append = False
                data = None
        
        if catagory == 'Ticker':
            data_to_append.insert(0, tickrs[i])
        else:  
            data_to_append.append(data)
            
    if data_to_append and do_append:
        big_data.append(data_to_append)
        
    time.sleep(1)
    

catagories[1], catagories[0] = catagories[0], catagories[1]
catagories.insert(2, 'time_series')
df = pd.DataFrame(big_data, columns=catagories)

#df = df.set_index('Ticker', inplace=True)

try:
    print(i)
    print(len(big_data))
    print(len(tickrs))
    df.to_csv('data\\ticker_info.csv', index=False)     
except:
    print(df)

