
import sys
import yfinance as yf
import numpy as np



def get_data(maskSize: int = 42, ticker: str = 'AAPL', 
            start_date: str = '1990-01-01', end_date: str = '2021-07-12') -> tuple:
    """This function retrieves the data for the experiment.
    It should return a numpy array with the input data."""
    # This function should call the get_data function and prepare the input data for the experiment
    # For now, we will just use the get_data function to get the data
    datapoints = yf.download(ticker, start_date, end_date)  # Get the data from the get_data function
    # print('datapoints', datapoints)
    # get the close prices from the data
    close_prices = datapoints['Close'].values  # Get the close prices from the data

    # get length of the close prices
    length = len(close_prices)
    
    # Prepare the input data for the experiment
    input_data = []  # Create an empty array to hold the input data
    reference_data = []
    for i in range(length - maskSize + 1):
        input_data.append(np.average(close_prices[i:i + maskSize])) # take average add other metrics later
        reference_data.append(close_prices[i:i + maskSize])  # Append the last value of the mask to the reference data

    # Convert the input data to a numpy array
    input_data = np.array(input_data, dtype=object)  # Convert the input data
    reference_data = np.array(reference_data, dtype=object)  # Convert the reference data

    # Return the input data
    return (input_data, reference_data)


def generate_agents(populationSize: int = 5, grounds: int = 1, options: int = 5, maskSize: int = 42) -> list:
    """This function generates a collective of agents of size populationSize.
    An agent is implemented as an array of weights ranging between 0 and 100"""
    collective = []
    for _ in range(populationSize):
        ws = np.random.rand(options, options, grounds+maskSize, 2)  # Randomly initialize the weight system
        ws = np.clip(ws, 0, 1)  # Ensure weights are between 0 and 1
        ws = 100 * ws  # Scale weights to a range of 0 to 100
        ws = np.round(ws,0)  # Round weights to the nearest integer
        collective.append(ws)

    # TODO: implement agent types: risk taking, risk averse, ...
    return collective


def simulate_experiment(collective: np.ndarray, data: np.ndarray) -> tuple:
    """This function runs the experiment. It takes the agents and averages and 
    calculates what each agent would do but also what the collective would do."""

    # TODO make sure that we get weights*datapoint for each datapoint
    colAgent = np.zeros((len(data),*collective[0].shape))  # Initialize the collective agent's weights
    weights = []
    for i in range(len(collective)):
        agentWeights = np.zeros((len(data[0]),*collective[0].shape))
        for j in range(len(data)):
            # For each agent, multiply its weights by the data point
            print(agentWeights)
            agentWeights[j,:,:,0,:] = collective[i][:,:,0,:]*data[0][j]
            print(agentWeights)
            agentWeights[j,:,:,1:,:] = collective[i][:,:,0:-1,:]*data[1][j]  # Multiply the agent's weights by the data point
            print(agentWeights)
            # print(colAgent[j])
            # print(colAgent[j].shape)/
            colAgent[j] += agentWeights[j]  # Add the agent's weights to the collective agent's weights
        weights.append(agentWeights)  # Append the agent's weights to the weights list

    weights.append(colAgent)
    return
    # print(weights[0].shape)

    # options = ['strong_buy', 'buy', 'hold', 'sell', 'strong_sell']
    options = 5
    cdec = []
    for x in range(len(collective)+1):
        dec = []
        weight_system = weights[x]
        de = np.ones(options) # Initialize a value array with 1s for each option, 1 means permitted and 0 means not permitted
        print(weight_system.shape)
        for y in range(len(data)):
            for option1 in range(options):
                for option2 in range(options):
                    if option1 != option2:
                        # print('competition test between', i, 'and', j)
                        # v i becomes not permitted (0) when it gets a -1 somewhere
                        # v j becomes not permitted (0) when it gets a -1 somewhere 
                        jwo1 = np.sum(weight_system[y,option1,option2,:,0])  # Justifying weight of option1 over option2
                        rwo1 = np.sum(weight_system[y,option1,option2,:,1])  # Requiring weight of option1 over option2
                        jwo2 = np.sum(weight_system[y,option2,option1,:,0])  # Justifying weight of option2 over option1
                        rwo2 = np.sum(weight_system[y,option2,option1,:,1])  # Requiring weight of option2 over option1

                        # print(jwo1, rwo1, jwo2, rwo2)

                        # The values are the sign of the difference between the justifying and requiring weights
                        v1 = np.sign(jwo1-rwo2)
                        v2 = np.sign(jwo2-rwo1)

                        if v1 == -1: de[option1] = 0
                        if v2 == -1: de[option2] = 0 

                        dec.append(de)
        print('decision', dec)
        cdec.append(dec)

    return (cdec, weights)


def evaluate(experiment: list) -> int:
    """ """

    return 1


def plot(evaluation: tuple, experiment: tuple, data: tuple) -> int:


    return 1

def main() -> int:
    """Main function to handle command line arguments."""

    data = get_data(maskSize=2, start_date = '2021-07-01', end_date = '2021-07-12')
    print('data', data[0], data[1])
    # return 1
    population = generate_agents(populationSize=1, maskSize=2)
    experiment = simulate_experiment(population, data)
    evaluation = evaluate(experiment)
    graph = plot(evaluation, experiment, data)

    return 1


if __name__ == '__main__':

    sys.exit(main())  
