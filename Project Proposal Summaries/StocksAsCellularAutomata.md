# Beating the stock market with a networked firm
## General Approach
1. Establish groupings (ie a firm) of nodes (brokers) with adjustable degrees of interconnectivity. Also establish interconnectivities between overall firms and sparse connections between brokers of other firms. Interconnectivity is defined as influence of broker on other broker's decision making. This will result in a network of clustered nodes with connections to the outside and some noise that will be treated as probability.
2. Each firm will have an exit timeline/time window in which the goal is to maximize profits and risk minimums and maximums. They will have access to the overall market data (ie the Dow Jones and the specific values of their stocks of interest).
3. The assumption will be that the firms will all be small enough to not change the overall market data over the previous 10-20 decades. Firms will compete with one another.
4. The firms will deal in individual stocks, not mutual funds.
5. Develop simple rules for each broker relating neighbor behavior, individual preference for risk (assume gaussian distribution from min to max within two standard deviations of mean within firm), global data, and timeline. Possibly use antifragile approach on some firms and set minimum range and maximum range for risk with gap in the middle.
6. Use data present from market to examine performance of each firm. Possibly examine some game theoretic balance of nash equilibrium for two firms choosing risk or safe.

## Benefits
Data is easily available for general performance of the market and plenty of stocks and their historical data can be randomly selected. There are a large number of parameters that can be adjusted at the individual level to examine different effects. 

## Downsides
Economic markets can be extremely volatile and this could still result in issues with fragility because there still is a limited quantity of data of market. Additionally, many directions already explored in this area because of the potential present. This may also not dictate an ideal strategy because if it were to accurately simulate the market, individual behaviors would change to account for it.

## Goals
1. Examine the impact of individual risk adversity within a grouping on profitability of the individual and the firm. Results will be viewed over various exit timelines and at different starting points in time.
2. Examine the value of making trading decisions based on the variable ratio between influence of neighboring firms and brokers' trades to market absolutes. The parameter in question here is the influence a given broker has. This may be adaptable. The market overall data can also be examined from the perspective of overall market or specific stocks.
3. Demonstrate value of antifragile approach to trading and examine game of choosing risky or safe investments against neighbors moves. (might not work since we can't actually drive the stock prices)

## Potentional Sources/Citations
- https://www.cs.cornell.edu/home/kleinber/networks-book/
- https://pubs.aeaweb.org/doi/pdfplus/10.1257/jel.20191434
- https://www.sciencedirect.com/science/article/pii/S0927539810000368
- https://www.sciencedirect.com/science/article/pii/S0957417421002414
- Antifragile (book by Nassim Taleb)

