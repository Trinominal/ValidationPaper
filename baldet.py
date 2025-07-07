
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
    """ """

    colAgent = np.zeros(*collective[0].shape)
    weights = []
    for i in range(len(collective)):
        agent = collective[i]
        colAgent += agent
        weights.append(agent*data)

    weights.append(colAgent*data)

    decisions = []
   
    return (decisions, weights)


def evaluate() -> tuple:
    """ """

    return (0)

def main() -> int:
    """Main function to handle command line arguments."""

    data = get_data()
    population = generate_agents()
    experiment = simulate_experiment(population[0], data[0])
    evaluation = evaluate(experiment)

    return 0


if __name__ == '__main__':

    sys.exit(main())  
