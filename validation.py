# echo.py

# import shlex
import sys
import numpy as np
import pandas as pd
# import edgar
# from edgar import *
# from edgar.xbrl import XBRLS
# import matplotlib.ticker as mtick
from datetime import timedelta
from datetime import datetime
import matplotlib.pyplot as plt
import mplfinance as mpf
import seaborn as sns
import yfinance as yf



def echo(phrase: str) -> None:
   """A dummy wrapper around print."""
   # for demonstration purposes, you can imagine that there is some
   # valuable and reusable logic inside this function
   print(phrase)


# for now I assume a context c is set fixed.
# also I assume that the options and grounds are fixed.
# what fixed means is that they are not changed during the execution of the program i.e. not more grounds or options are added.
# this is a simplification for now, but it should be easy to change later on, if needed.
options = np.zeros(5) # strong sell, sell, hold, buy, strong buy
grounds = np.zeros(10) 


########## WEIGHT SYSTEM FUNCTIONS ####################################################################################################################################

def generateWeightSystem(options, grounds) :
    """This function should generate a weight system with more interesting weights than only 1s.
    This can potentially be guided by axioms or other logic."""
    ws = instantiateWeightSystem(options, grounds)

    ws = np.random.rand(*ws.shape)  # Randomly initialize the weight system
    ws = np.clip(ws, 0, 1)  # Ensure weights are between 0 and 1
    ws = 100 * ws  # Scale weights to a range of 0 to 100
    
    return ws


def instantiateWeightSystem(options, grounds) :
    """This function instanties a weight system initialized with a choice set and a set of grounds."""
    # We encode it such that the ground is pro option1 and con option2 i.e. ws[option1, option2, ground, 0] is the justifying weight of option1 over option2
    # a weight of 0,0 means that the ground is not relevant for the comparison or that the ws is not familiar with the ground

    lc = len(options)
    lr = len(grounds)
    ws = np.ones((lc, lc, lr, 2))

    return ws


def setWeight(weight_system: dict, option1: str, option2: str, ground: str, weight1: float, weight2: float) -> None:
    """This function sets a value for a ground in the weight system."""
    # the weight system is an tensor of shape (options, options, grounds, 2)
    # where the first two dimensions are the options and the last dimension is the grounds
    # weight '0' is the justifying weight and weight '1' is the requiring weight

    weight_system[option1][option2][ground][0] = weight1
    weight_system[option1][option2][ground][1] = weight2


def addWeightSystems(weight_systems: list) :
    """This function combines a list of weights systems into a single weight system."""
    aws = np.zeros_like(weight_systems[0])
    
    for ws in weight_systems:
        aws = np.add(aws, ws)
    
    return aws


def agentType(weight_system: np.ndarray, type: str = 'default') -> np.ndarray:
    """This function adds a bias to the weight system. The bias is added to the justifying weights."""
    # The bias is an  that is added to the justifying weights of the weight system
    # The requiring weights are not changed

    WS = weight_system.copy() # Create a copy of the weight system to avoid modifying the original. Not sure if this is needed, but it is safer. Maybe a deep copy is needed.

    if type == 'default':
        return WS
    elif type == 'riskTaking':
        # Risk taking agents do ?
    elif type == 'riskAverse':
        # Risk averse agents do ?
    elif type == 'optimistic':
        # Optimistic agents do ?
    elif type == 'pessimistic':
        # Pessimistic agents do ?
    elif type == 'conformist':
        # Conformist agents do ?
    elif type == 'independent':
        # Independent agents do ?


    # TODO: Select correct option for weight change
    elif type == 'selling':
        # Selling agents are agents that prefer the option sell.
        # increase the justifying weight of the option sell
        WS[0, :, :, 0] += 10
        WS[:, 0, :, 0] += 10
        # increase the requiring weight of the option sell
        WS[0, :, :, 1] += 5
        WS[:, 0, :, 1] += 5
        return WS
    elif type == 'buying':
        # Buying agents are agents that prefer the option buy.
        # increase the justifying weight of the option buy
        WS[1, :, :, 0] += 10
        WS[:, 1, :, 0] += 10
        # increase the requiring weight of the option buy
        WS[1, :, :, 1] += 5
        WS[:, 1, :, 1] += 5
        return WS
    elif type == 'stronglySelling':
        # Strongly selling agents are agents that strongly prefer the option Strong sell.
        # increase the justifying weight of the option strong sell
        WS[2, :, :, 0] += 10
        WS[:, 2, :, 0] += 10
        # increase the requiring weight of the option strong sell
        WS[2, :, :, 1] += 5
        WS[:, 2, :, 1] += 5
        return WS
    elif type == 'stronglyBuying':
        # Strongly buying agents are agents that strongly prefer the option Strong buy.
        # increase the justifying weight of the option strong buy
        WS[3, :, :, 0] += 10
        WS[:, 3, :, 0] += 10
        # increase the requiring weight of the option strong buy
        WS[3, :, :, 1] += 5
        WS[:, 3, :, 1] += 5
        return WS
    elif type == 'hold':
        # Neutral agents are agents that do not prefer the option hold
        # increase the justifying weight of the option hold
        WS[4, :, :, 0] += 10
        WS[:, 4, :, 0] += 10
        # increase the requiring weight of the option hold
        WS[4, :, :, 1] += 5
        WS[:, 4, :, 1] += 5
        return WS
    else:
        raise ValueError("Unknown agent type")
    
    return WS


def validateWeightSystem(weight_system, type: str = 'default') -> np.ndarray:
    """There may be some logic to validate the weight system. This function applies the contraints to the weight system to make it valid."""
    validWS = weight_system.copy() # Create a copy of the weight system to avoid modifying the original. Not sure if this is needed, but it is safer. Maybe a deep copy is needed.
    # type can be 'default', 'strict', 'singleProp', 'probabilistic', 'delta', 'uniform'

    # Add your validation logic here
    if type == 'default': 
        # Ensure all weights are non-negative
        validWS[validWS < 0] = 0
        return validWS
    elif type == 'strict':
        # Ensure all weights are between 0 and 100
        validWS[validWS < 0] = 0
        validWS[validWS > 100] = 100
        return validWS
    elif type == 'singleProp':
        # Ensure that the justifying weight is equal to the requiring weight for each ground
        for i in range(weight_system.shape[0]):
            for j in range(weight_system.shape[1]):
                for k in range(weight_system.shape[2]):
                    if weight_system[i, j, k, 0] != weight_system[i, j, k, 1]:
                        # If they are not equal, set them to the average
                        avg = (weight_system[i, j, k, 0] + weight_system[i, j, k, 1]) / 2
                        validWS[i, j, k, 0] = avg
                        validWS[i, j, k, 1] = avg
        return validWS
    elif type == 'probabilistic':
        # Ensure that the weights are probabilities, i.e., they sum to 1 for each ground
        # assume sigma additivity amongst grounds. i.e. grounds are independent. Not sure if how sensible this assumption is.
        # Not clear yet how this works with the justifying and requiring weights.
        # which values should add up to 1?
        # validWS[validWS < 0] = 0
        # for i in range(weight_system.shape[0]):
        #     for j in range(weight_system.shape[1]):
        #         for k in range(weight_system.shape[2]):
                    # total = weight_system[i, j, k, 0] + weight_system[i, j, k, 1]
                    # if total > 0:
                    #     validWS[i, j, k, 0] /= total
                    #     validWS[i, j, k, 1] /= total
        return validWS
    elif type == 'delta':
        # Not exactly sure how to put this. The idea behind this is that a reason has weight such that the ground is only a reason for on of the options.
        # not sure if a ground can have justifying weight for both options.
        # It is easy to see that justifying weight for o1 and requiring weight of o2 counteract. so only the delta counts anyway
        # Ensure that if they both have a weight, one of them is set to zero.
        # this does not account for the possiblity of w(g,o1) = (1,0), w(g,o2) = (1,0) i.e. allows it
        for i in range(weight_system.shape[0]):
            for j in range(weight_system.shape[1]):
                for k in range(weight_system.shape[2]):
                    if weight_system[i, j, k, 0] != 0 and weight_system[j, i, k, 0] != 0:
                        # If both weights are non-zero, set one of them to zero
                        if weight_system[i, j, k, 0] > weight_system[j, i, k, 1]:
                            validWS[i, j, k, 0] = weight_system[i, j, k, 0] - validWS[j, i, k, 1]
                            validWS[j, i, k, 1] = 0
                        elif weight_system[i, j, k, 0] < weight_system[j, i, k, 1]:
                            validWS[j, i, k, 1] = weight_system[j, i, k, 1] - validWS[i, j, k, 0]
                            validWS[i, j, k, 0] = 0
                        elif weight_system[i, j, k, 0] == weight_system[j, i, k, 0]:
                            # If they are equal, set both to zero
                            validWS[i, j, k, 0] = 0
                            validWS[j, i, k, 1] = 0
        return validWS
    elif type == 'uniform':
        return np.ones_like(weight_system)
    else:
        return -1 # or raise an exception


################ SCALE FUNCTIONS ############################################################################################################################################


def fivePointScale(wdmin: int, wmin: int, wneut: int, wplus: int, wdplus: int) -> int:
    """This function takes a value and returns a value on a 5-point scale."""
    # The scale is: --, -, 0, +, ++
    plus = 2*wdplus + wplus 
    min = 2*wdmin + wmin
    value = plus - min  # Calculate the value based on the weights

    if 2*abs(value) < wneut:
        return 0
    elif value < 0:
        if wdmin > wmin:
            return -2
        elif wmin > wdmin:
            return -1 
    elif value > 0:
        if wdplus > wplus:
            return 2
        elif wplus > wdplus:
            return 1
    else:
        return 0


def detachment(weight_system, option1: int, option2: int) -> tuple:
    """Takes a weight system and 2 options; then outputs two values, one for each option."""
    # This function should calculate the detachment value for the two options

    jwo1 = np.sum(weight_system[option1,option2,:,0])  # Justifying weight of option1 over option2
    rwo1 = np.sum(weight_system[option1,option2,:,1])  # Requiring weight of option1 over option2
    jwo2 = np.sum(weight_system[option2,option1,:,0])  # Justifying weight of option2 over option1
    rwo2 = np.sum(weight_system[option2,option1,:,1])  # Requiring weight of option2 over option1

    # print(jwo1, rwo1, jwo2, rwo2)

    # The values are the sign of the difference between the justifying and requiring weights
    v1 = np.sign(jwo1-rwo2)
    v2 = np.sign(jwo2-rwo1)

    return v1, v2


def competition(ws, options) -> None:
    """Takes a weight systems and choice set containing options; 
    then runs a pairwise competition between all options using detachment() 
    and outputs a value for each option in the choice set."""
    # also known as dynamic scale

    v = np.ones(len(options))
    for i in range(len(options)):
        for j in range(len(options)):
            if i != j:
                print(i,j)
                v1, v2 = detachment(ws, options[i], options[j])
                v[i] = 0 if v1 == -1 else 1
                v[j] = 0 if v2 == -1 else 1
                # Here we would do something with v1 and v2, like updating a score or similar
                # For now, we just print them
                print(f"Detachment between {options[i]} and {options[j]}: {v1}, {v2}")

    return v


############### METRICS FUNCTIONS ####################################################################################################################################

def metric1(iValues, cValues) -> float:
    """This function calculates a metric between two sets of values."""
    # Loss(R, c, RR, O)(I) = Sum_o∈O Sum_i∈I [D(R, wi, c, RR, O)(o)̸ = D(R, wI , c, RR, O)(o)]
    # Assuming isValues and cValues are lists or arrays of values
    # Here we would calculate the loss metric for each value in isValues
    length = len(iValues) if len(iValues) == len(cValues) else -1

    a = np.array(iValues)
    b = np.array(cValues)
    disagrees = (a != b) if length > 0 else -1
    print(f"Loss: {disagrees}")

    return disagrees


def metric2(iValues, cValues) -> float:
    """This function calculates a metric between two sets of values."""
    # coherence = Sum_o∈O [P (x, y) ∈ I × I1{D(R,wx,c,RR,O)(o)}D(R, wy , c, RR, O)(o)]
    # Assuming isValues and cValues are lists or arrays of values
    # Here we would calculate the loss metric for each value in isValues
    length = len(iValues) if len(iValues) == len(cValues) else -1

    a = np.array(iValues)
    b = np.array(cValues)
    agrees = (a == b) if length > 0 else -1
    print(f"Agrees: {agrees}")

    return agrees
    

def metric3(isValues, cValues) -> float:
    """This function calculates a metric between two sets of values."""
    # agreement = Sum_o∈O [Prod_i∈I 1{D(R,wa1 ,c,RR,O)(o)}D(R, wi, c, RR, O)(o)]

    agreement = np.zeros_like(isValues)

    for i in range(len(isValues)):
        # Assuming isValues and cValues are lists or arrays of values
        # Here we would calculate the loss metric for each value in isValues
        agreement[i] = metric2(isValues[i], cValues)  # Call metric1 to get the loss for the current iValues

    print(f"Agree's: {agreement}")

    return agreement


def metric4(isValues, cValues) -> float:
    """This function calculates a metric between two sets of values."""
    # This is a placeholder for now, as the actual implementation of the metric is not yet defined
    # This does the opposite of metric3, i.e. it calculates the disagreement

    disagreement = np.zeros_like(isValues)

    for i in range(len(isValues)):
        # Assuming isValues and cValues are lists or arrays of values
        # Here we would calculate the loss metric for each value in isValues
        disagreement[i] = metric1(isValues[i], cValues)  # Call metric1 to get the loss for the current iValues

    print(f"Disagree's: {disagreement}")

    return disagreement


def Metricwrapper(metric: str, isValues, cValues) -> None:
    """This function is a wrapper for the loss metrics.
    It should take the values from the weight system and the competition and return a value."""
    # This is a placeholder for now, as the actual implementation of the metrics is not yet defined
    # The metric can be 'loss1', 'loss2', 'loss3'.

    total = 0  # Initialize loss to zero
    tmp = []  # Temporary variable to hold the loss for each index

    if metric == 'disagrees':
        # Call lossMetric1 with the appropriate values
        for iValues in isValues:
            tmp = metric1(iValues, cValues) 
            total += tmp.sum()
        # After the loop, print the total loss
        print(f"Total loss: {total}")

    elif metric == 'agrees':
        # Call lossMetric2 with the appropriate values
        for iValues in isValues:
            tmp = metric2(iValues, cValues)   
            total += tmp.sum()
        # After the loop, print the total loss
        print(f"Total agrees: {total}")

    elif metric == 'agreement':
        # Call lossMetric3 with the appropriate values
        disagreement = np.zeros(len(cValues))  # Initialize agreement array
        # Iterate over each isValue and calculate the agreement for each cValue
        tmp = metric3(isValues, cValues)
        for j in range(len(cValues)):
            disagreement[j] = (len(isValues) == np.sum(tmp[:, j]))  # Sum the losses for each cValue
        total = np.sum(disagreement)  # Sum the agreement values
        print(f"Total agreement: {total}")

    elif metric == 'disagreement':
        # Call lossMetric4 with the appropriate values 
        # All individuals disagree with the collective on the state of an option
        agreement = np.zeros(len(cValues))  # Initialize agreement array
        # Iterate over each isValue and calculate the agreement for each cValue
        tmp = metric4(isValues, cValues)
        for j in range(len(cValues)):
            agreement[j] = (len(isValues) == np.sum(tmp[:, j]))  # Sum the losses for each cValue
        total = np.sum(agreement)  # Sum the agreement values
        print(f"Total disagreement: {total}")

    else:
        raise ValueError("Unknown metric type")



############### DATA RETRIEVAL FUNCTIONS ############################################################################################################################################

def get_data():
    
    # Set the start and end date
    start_date = '1990-01-01'
    end_date = '2021-07-12'

    # Set the ticker
    # ticker = 'AMZN'
    ticker = 'AAPL'
    # ticker = 'VFIAX'

    # Get the data
    data = yf.download(ticker, start_date, end_date)

    # Print 5 rows
    print(data)

    return data


def get_data2():

    # set the start and end dates for our market data request to be TTM
    end_date = datetime(year=2025, month=3, day=1)
    start_date = end_date - timedelta(days=5*365)

    # set the name of the ticker we want to download market data for
    ticker = "ASML"  # AEX index on Euronext Amsterdam

    # download market data
    df = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False
    )

    # restructure the default multi-index dataframe to our preferred format
    df = df.stack(level="Ticker", future_stack=True)
    df.index.names = ["Date", "Symbol"]
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.swaplevel(0, 1)
    df = df.sort_index()
    print(df)

    # initialize an empty figure
    plt.figure(figsize=(10, 6))
    plt.grid(alpha=0.5)

    # plot the closing prices
    plt.plot(
        df.xs(ticker).index,
        df.xs(ticker)["Close"],
        color="blue",
        linewidth=1.5
    )

    # set the plot title and axis labels
    plt.title(f"{ticker} Closing Price [TTM]")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")

    # finish constructing the plot
    plt.xticks(rotation=45)
    plt.tight_layout()

    # show the plot
    plt.show()

    # plot the open, high, low, and close candlestick data
    mpf.plot(
        df.xs(ticker),
        type="candle",
        style="yahoo",
        figsize=(14, 7),
        volume=True,
        title=f"{ticker} - Basic Candlestick Chart"
        
    )
    mpf.show()



########### TEST FUNCTIONS ############################################################################################################################################


def test(x: int) -> None:
    # y = np.zeros((3,3,5,2))
    # print(y)
    # print(np.sign(0.3))
    # y1 = generateWeightSystem(options, grounds)
    # y2 = instantiateWeightSystem(options, grounds)
    # y = addWeightSystems([y1, y2])
    # print(y1)
    # print(np.sum(y[0,1,:,0]))
    # z = detachment(y, 0, 1)
    # print(z)
    # print(np.array([1, 2, 3]) == np.array([1, 2, 3]))
    # print(np.array([1, 2, 3]) == np.array([1, 2, 4]))

    # print((np.array([1, 2, 3]) == np.array([1, 2, 3])) == (np.array([1, 2, 3]) == np.array([1, 2, 4])))
    # print(np.array([1, 2, 3]) == (np.array([1, 2, 3]) == np.array([1, 2, 4])))

    # print(np.array([1, 1, 1]) == (np.array([1, 1, 1]) == np.array([1, 1, 1])))

    # print(metric1([1, 2, 3], [1, 2, 3]))
    # print(metric1([1, 2, 3], [1, 2, 4]))

    # print(metric2([[1, 2, 3]], [1, 2, 3]))
    # print(metric2([[1, 2, 3]], [1, 2, 4]))

    # print(metric3([[1, 1, 1],[1, 1, 0]], [1, 1, 1]))
    # print(metric3([[1, 1, 1],[1, 1, 0],[1, 0, 1]], [1,1,1]))

    # print(metric4([[1, 1, 1],[1, 1, 0]], [1, 1, 1]))
    # print(metric4([[1, 1, 1],[1, 1, 0],[1, 0, 1]], [1,1,1]))

    print("----------------")

    wrapperMetric('disagrees', [[1, 1, 1]], [1, 1, 1])
    print("----")
    wrapperMetric('disagrees', [[1, 1, 1]], [1, 1, 0])
    print("----------------")

    wrapperMetric('agrees', [[1, 1, 1]], [1, 1, 1])
    print("----")
    wrapperMetric('agrees', [[1, 1, 1]], [1, 1, 0])
    print("----------------")

    wrapperMetric('agreement', [[1, 1, 1],[1, 1, 1]], [1, 1, 1])
    print("----")
    wrapperMetric('agreement', [[1, 1, 1],[1, 1, 0]], [1, 1, 0])
    print("----------------")

    wrapperMetric('disagreement', [[1, 1, 1],[1, 1, 0]], [1, 1, 1])
    print("----")
    wrapperMetric('disagreement', [[1, 1, 1],[1, 1, 1]], [1, 1, 0])
    print("----------------")

    pass


########### PLOTTING FUNCTIONS ############################################################################################################################################



########### MAIN FUNCTION #################################################################################################################################################


def main() -> int:
    """Main function to handle command line arguments."""
    # competition(instantiateWeightSystem(options, grounds), range(len(options)))
    # test(1)
    # getDataSet()
    # balance_sheet = Company("AAPL").get_financials().balance_sheet()         
    # print(balance_sheet)
    # getDataSet3()

    # plot_revenue("AAPL")
    # portfolio()
    get_data2()
    return 0

if __name__ == '__main__':
    

    # set_identity('Vincent de Wit vincent.j.wit@gmail.com')


    sys.exit(main())  




    # TODO: 
    # - How are the 5 options operationalized? i.e. how are they defined?
    # - What are usefull metrics to determine the "quality" of the current state and a possible next state? 
    #       Where next state is the state after a decision(option has been chosen) has been made.
    # - What are usefull metrics to compare the weight systems?
    # - How to set the remaining agent type weight systems?
    #       Maybe Benoit, David and Aleks paper is a way to go.
