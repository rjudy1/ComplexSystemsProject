# Stock Analysis as Cellular
## General Approach
Split individual brokers into groupings/firms and make decisions based on neighbors' actions within the grouping as well as a sparse randomized connection to brokers in other firms/outside informants/overall group/firm knowledge. This will result in a network of clustered nodes with connections to the outside and some noise that will be treated as probability. Try to develop simple rules for each broker to simulate overall market resembling market shifts over some interval, perhaps 2-50 years. Add variability such as risk adversity in trades, target exit timeline, length of time required to hold trades, and antifragility approach for different brokers. Use data available of stock positions at some given time to start the process, seeing how each results and how the market value is affected. This will isolate world events from the market but might demonstrate the bottom-up results are of higher significance.

## Benefits
Data is easily available for general performance of the market and plenty of stocks and their historical data can be randomly selected. There are a large number of parameters that can be adjusted at the individual level to examine different effects.

## Downsides
Economic markets can be extremely volatile and this could still result in issues with fragility because there still is a limited quantity of data of market. Additionally, many directions already explored in this area because of the potential present. This may also not dictate an ideal strategy because if it were to accurately simulate the market, individual behaviors would change to account for it.

## Overall Goal
To simulate the movement of the Dow Jones or some other market metrics over time based on the actions of individual nodes and their neighbors' behaviors.


## Potentional Sources/Citations
https://www.cs.cornell.edu/home/kleinber/networks-book/
