# Examing the impact of antifragility and networked based decision making on the stock market
## Brief description
Using some of the theories from Taleb regarding antifragility, construct a network of firms and networked brokers inside firms to explore the impact of different preferences for risk as well as impact of weighting neighbors actions in decisions.

## Goals
1. Examine the impact of individual risk adversity within a grouping on profitability of the individual and the firm. Approaches will include an antifragile approach. Results will be viewed over various exit timelines and at different starting points in time.
2. Examine the value of making trading decisions based on the variable ratio between influence of neighboring firms and brokers' trades to market absolutes. The parameter in question here is the influence a given broker has. This may be adaptable. The market overall data can also be examined from the perspective of overall market or specific stocks.

## Motivation
Economic markets are extremely volatile and present an active area of research. This project has the advantages of readily available historical data and a large number of parameters able to be examined at different levels. Determining optimal strategies both at the firm level and at the individual level would be highly beneficial in the real world.

## Brief Background
In Antifragile, Nassim Taleb discussed designing systems that benefit from shocks; these systems will improve when volatile black swan events occur. These events represent randomness and uncertainty, the sort of events that have caused massive crashes and growths in the stock market such as in the 2008 housing crisis and the 2020 pandemic. A firm attempting to make money in the stock market would attempt to have investments that are not only robust and able to remain the same; it would attempt to invest in such a way as to make money at a steady rate but also to benefit from major events and grow stronger from them. Previous attempts to model the market include modeling of the relationships between stocks and locating key stocks to represent the overall market, predicting the market using CNNs and deep learning, examing how network effects shift the market equilibrium, and studying the presence of power law effects in the market. Financial advisors tend to try to spread investments based on exit timeline, often using a small amount of high cap and low cap stocks with the largest percentage of funds in mid cap and fairly safe stocks. These techniques provides openings for studying the relationship of neighboring influence and profitability when buying stocks as well as examining the results of approaches modeling different levels of risk in selecting investments.

## General Approach
1. Establish groupings (ie a firm) of nodes (brokers) with adjustable degrees of interconnectivity. Also establish interconnectivities between overall firms and sparse connections between brokers of other firms. Interconnectivity is defined as influence of broker on other broker's decision making. This will result in a network of clustered nodes with connections to the outside and some noise that will be treated as probability.
2. Each firm will have an exit timeline/time window in which the goal is to maximize profits and risk minimums and maximums. They will have access to the overall market data (ie the Dow Jones and the specific values of their stocks of interest).
3. The assumption will be that the firms will all be small enough to not change the overall market data over the previous 10-20 decades. Firms will compete with one another.
4. The firms will deal in individual stocks, not mutual funds.
5. Develop simple rules for each broker relating neighbor behavior, individual preference for risk (assume gaussian distribution from min to max within two standard deviations of mean within firm), global data, and timeline. Possibly use antifragile approach on some firms and set minimum range and maximum range for risk with gap in the middle.
6. Use data present from market to examine performance of each firm. Possibly examine some game theoretic balance of nash equilibrium for two firms choosing risk or safe approach.


## Downsides
Economic markets can be extremely volatile and this could still result in issues with fragility because there still is a limited quantity of data of market. Additionally, many directions already explored in this area because of the potential present. This may also not dictate an ideal strategy because if it were to accurately simulate the market, individual behaviors would change to account for it.


## Citations
- https://www.cs.cornell.edu/home/kleinber/networks-book/, particularly chapters 9-12, 17, 22
- https://pubs.aeaweb.org/doi/pdfplus/10.1257/jel.20191434
- https://www.sciencedirect.com/science/article/pii/S0927539810000368
- https://www.sciencedirect.com/science/article/pii/S0957417421002414
- Antifragile (book by Nassim Taleb)
- dates noted as stock crashes (https://en.wikipedia.org/wiki/List_of_stock_market_crashes_and_bear_markets)

