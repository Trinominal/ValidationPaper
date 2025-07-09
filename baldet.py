
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

    # get the close prices from the data
    close_prices = datapoints['Close'].values  # Get the close prices from the data

    # get length of the close prices
    length = len(close_prices)
    
    # Prepare the input data for the experiment
    input_data = []  # Create an empty array to hold the input data
    reference_data = []
    for i in range(length - maskSize):
        input_data.append(np.average(close_prices[i:i + maskSize])) # take average add other metrics later
        reference_data.append(close_prices[i:i + maskSize])  # Append the last value of the mask to the reference data

    # Convert the input data to a numpy array
    input_data = np.array(input_data, dtype=object)  # Convert the input data
    reference_data = np.array(reference_data, dtype=object)  # Convert the reference data

    # Return the input data
    return (input_data, reference_data)


def generate_agents(populationSize: int = 5, grounds: int = 1, options: int = 5) -> tuple:
    """This function generates a collective of agents of size populationSize.
    An agent is implemented as an array of weights ranging between 0 and 100"""
    collective = []
    for i in range(populationSize):
        ws = np.random.rand((options, options, grounds, 2))  # Randomly initialize the weight system
        ws = np.clip(ws, 0, 1)  # Ensure weights are between 0 and 1
        ws = 100 * ws  # Scale weights to a range of 0 to 100
        collective.append(ws)
       
    # TODO: implement agent types: risk taking, risk averse, ...
    return (collective)


def simulate_experiment(collective: np.ndarray, data: np.ndarray) -> tuple:
    """This function runs the experiment. It takes the agents and averages and 
    calculates what each agent would do but also what the collective would do."""

    # TODO make sure that we get weights*datapoint for each datapoint
    colAgent = np.zeros(*collective[0].shape)
    weights = []
    for i in range(len(collective)):
        agent = collective[i]
        colAgent += agent
        weights.append(agent*data)

    weights.append(colAgent*data)

    # options = ['strong_buy', 'buy', 'hold', 'sell', 'strong_sell']
    options=5
    dec = []
    for x in range(len(collective+1)):
        weight_system = weights[x]
        de = np.ones(len(options)) # Initialize a value array with 1s for each option, 1 means permitted and 0 means not permitted
        for option1 in range(len(options)):
            for option2 in range(len(options)):
                if option1 != option2:
                    # print('competition test between', i, 'and', j)
                    # v i becomes not permitted (0) when it gets a -1 somewhere
                    # v j becomes not permitted (0) when it gets a -1 somewhere 
                    jwo1 = np.sum(weight_system[option1,option2,:,0])  # Justifying weight of option1 over option2
                    rwo1 = np.sum(weight_system[option1,option2,:,1])  # Requiring weight of option1 over option2
                    jwo2 = np.sum(weight_system[option2,option1,:,0])  # Justifying weight of option2 over option1
                    rwo2 = np.sum(weight_system[option2,option1,:,1])  # Requiring weight of option2 over option1

                    # print(jwo1, rwo1, jwo2, rwo2)

                    # The values are the sign of the difference between the justifying and requiring weights
                    v1 = np.sign(jwo1-rwo2)
                    v2 = np.sign(jwo2-rwo1)

                    if v1 == -1: de[option1] = 0
                    if v2 == -1: de[option2] = 0 

                    dec.append(de)
   
    return (dec, weights)


def evaluate() -> tuple:
    """ """

    return (0)


def plot():


    pass

def main() -> int:
    """Main function to handle command line arguments."""

    data = get_data()
    population = generate_agents()
    experiment = simulate_experiment(population[0], data[0])
    evaluation = evaluate(experiment)

    return 0


if __name__ == '__main__':

    sys.exit(main())  
