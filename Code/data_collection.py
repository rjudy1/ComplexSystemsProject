import yfinance as yf
import matplotlib.pyplot as plt
import csv
import time



def generate_combinations():
    l = []
    for length in range(1, 4):  # Generate combinations of length 2, 3, and 4
        for combo in generate_combinations_of_length(length):
            l.append([combo])
    return l

def generate_combinations_of_length(length):
    if length == 1:
        return [chr(ord('A') + i) for i in range(26)]
    else:
        shorter_combos = generate_combinations_of_length(length - 1)
        return [combo + chr(ord('A') + i) for combo in shorter_combos for i in range(26)]

possible_tickers = generate_combinations()

# name of csv file
filename = "valid_tickers.csv"
 
# writing to csv file
with open(filename, 'w', newline='') as csvfile:
    # creating a csv dict writer object
    writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
 
    for symbol in possible_tickers:
        ticker = yf.Ticker(symbol[0])
        data = ticker.history(start='2006-01-01', end='2009-01-01', interval="1d")
        if len(data) != 0:
            writer.writerow(symbol)
            
        time.sleep(1.8)

# This works but only for data in 2016
# indices = []
# with open('data\\yahoo_tickers_in_2016.csv') as csvfile:
#     reader = csv.reader(csvfile, delimiter=',')
#     for i, row in enumerate(reader):
#         indices.append(row[0])
        
# print(indices)

# data = yf.download("APPL", start="2020-01-01", end="2021-01-05")
# data['Close'].plot()
# plt.show()
 
 
