# Project Planning
class Broker {
    Id/Hashkey
    Influence
    Risk aversity
    Memory
    Rules
    Adaptability
    Affiliation? instead of firm class
    Owned stocks
    Profit margin
    Money available
}

class Firm {
    Grid<Row<Broker>>
    Dimensions
    Edges
    Exterior Accessible Parameters 
        Influence
        timescale?
        Information access
        Overall adaptability?
}

class Simulation
    List<Brokers>
    Timeline
    Evolvability


// use graphviz to build visualizations?

How are we going to quantify individual attributes?

What exactly will connections mean?
- score every trade option
- influence/amount of influence? quantify with neighbors certainty or action?
- greatest amount within firm
- how will we choose which brokers connect to others? from within firm and then random to others, concentrate based on network theory

What equations/parameters will we need?
- A high/low margin for sell/buy from global data
- an adjustment term based on the sum of influence/input and decision of the neighbors
- use volume data to create some firms resembling overall decisions made irl
- Influence attribute for a broker can increase as margins increase while simulatenously decreasing how much influence others have on him?


How many different stocks to choose from? 
- we need to make sure to allow stocks that went under but were available at the start of our simulation

Useful Tools
- Python3
- graphviz
- 