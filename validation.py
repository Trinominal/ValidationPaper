# echo.py

import shlex
import sys
import numpy as np
import pandas as pd

def echo(phrase: str) -> None:
   """A dummy wrapper around print."""
   # for demonstration purposes, you can imagine that there is some
   # valuable and reusable logic inside this function
   print(phrase)


# for now I assume a context c is set fixed.
# also I assume that the choices and grounds are fixed.
# what fixed means is that they are not changed during the execution of the program i.e. not more grounds or choices are added.
# this is a simplification for now, but it should be easy to change later on, if needed.
choices = np.zeros(3)
grounds = np.zeros(5)


def generateWeightSystem(choices, grounds) :
    """This function should generate a weight system with more interesting weights than only 1s.
    This can potentially be guided by axioms or other logic."""
    ws = instantiateWeightSystem(choices, grounds)

    ws = np.random.rand(*ws.shape)  # Randomly initialize the weight system
    ws = np.clip(ws, 0, 1)  # Ensure weights are between 0 and 1    
    ws = 100 * ws  # Scale weights to a range of 0 to 100
    
    return ws


def instantiateWeightSystem(choices, grounds) :
    """This function instanties a weight system initialized with a choice set and a set of grounds."""
    # We encode it such that the ground is pro option1 and con option2 i.e. ws[option1, option2, ground, 0] is the justifying weight of option1 over option2
    # a weight of 0,0 means that the ground is not relevant for the comparison or that the ws is not familiar with the ground

    lc = len(choices)
    lr = len(grounds)
    ws = np.ones((lc, lc, lr, 2))

    return ws


def setWeight(weight_system: dict, option1: str, option2: str, ground: str, weight1: float, weight2: float) -> None:
    """This function sets a value for a ground in the weight system."""
    # the weight system is an tensor of shape (choices, choices, grounds, 2)
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


def validateWeightSystem(weight_system, type: str = 'default') -> np.ndarray:
    """There may be some logic to validate the weight system. This function applies the contraints to the weight system to make it valid."""
    validWS = weight_system.copy()
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


def wrapperMetric(metric: str, isValues, cValues) -> None:
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

# Get / Generate data set for testing
def getDataSet() -> tuple:
    """This function returns a data set for testing."""
    # df1 = pd.read_csv('out_ticks.csv')
    # df2 = pd.read_csv('out_ohlcv.csv')
    df1 = pd.read_csv('out_ticks.csv', index_col=0)
    df2 = pd.read_csv('out_ohlcv.csv', index_col=0)
    print(df1)
    print(df2)
    return df1, df2








def test(x: int) -> None:
    # y = np.zeros((3,3,5,2))
    # print(y)
    # print(np.sign(0.3))
    # y1 = generateWeightSystem(choices, grounds)
    # y2 = instantiateWeightSystem(choices, grounds)
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





def main() -> int:
    """Main function to handle command line arguments."""
    # competition(instantiateWeightSystem(choices, grounds), range(len(choices)))
    test(1)
    # getDataSet()
    return 0

if __name__ == '__main__':
    sys.exit(main())  