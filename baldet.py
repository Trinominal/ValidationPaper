
import sys
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt



def get_data(maskSize: int = 42, ticker: str = 'AAPL', 
            start_date: str = '1990-01-01', end_date: str = '2021-07-12') -> tuple:
    """This function retrieves the data for the experiment.
    It should return a numpy array with the input data."""
    # This function should call the get_data function and prepare the input data for the experiment
    # For now, we will just use the get_data function to get the data
    datapoints = yf.download(ticker, start_date, end_date)  # Get the data from the get_data function
    # print('datapoints', datapoints)
    # get the close prices from the data
    close_prices = [x for xs in datapoints['Close'].values for x in xs] # Get the close prices from the data

    # get length of the close prices
    length = len(close_prices)
    
    # Prepare the input data for the experiment
    input_data = []  # Create an empty array to hold the input data
    reference_data = []
    for i in range(length - maskSize + 1):
        riskAdjustedReturn = 1
        average = np.average(close_prices[i:i + maskSize])
        input_data.append([average, riskAdjustedReturn]) # take average add other metrics later

        reference_data.append(close_prices[i:i + maskSize])  # Append the last value of the mask to the reference data
        
    # Convert the input data to a numpy array
    input_data = np.array(input_data, dtype=object)  # Convert the input data
    reference_data = np.array(reference_data, dtype=object)  # Convert the reference data

    # Return the input data
    return (input_data, reference_data)


def generate_agents(populationSize: int = 5, grounds: int = 2, options: int = 2, maskSize: int = 42, agentTypes = ['random', 'random', 'random', 'random', 'ones']) -> list:
    """This function generates a collective of agents of size populationSize.
    An agent is implemented as an array of weights ranging between 0 and 100"""
    # There is at the moment only one ground, the average. Next to that each value in the mask is a ground.
    # we get an array of options x options x grounds x 2
    # where the first dimension is the justifying weight and the second dimension is the requiring weight
    collective = []
    for _ in range(populationSize):
        agentType = agentTypes[_]
        if agentType == 'random':
            ws = np.random.rand(options, options, grounds+maskSize, 2)  # Randomly initialize the weight system
            ws = np.clip(ws, 0, 1)  # Ensure weights are between 0 and 1
            ws = 100 * ws  # Scale weights to a range of 0 to 100
            ws = np.round(ws,0)  # Round weights to the nearest integer
        elif agentType == 'ones':
            ws = np.ones((options, options, grounds+maskSize, 2)) 
        collective.append(ws)

    # TODO: implement agent types: risk taking, risk averse, ...
    return collective


def simulate_experiment(collective: np.ndarray, data: np.ndarray) -> tuple:
    """This function runs the experiment. It takes the agents and averages and 
    calculates what each agent would do but also what the collective would do."""

    # print(data[0].size, data[0].shape)
    # print(data[1].size, data[1].shape)

    # TODO make sure that we get weights*datapoint for each datapoint
    colAgent = np.zeros((len(data[0]),*collective[0].shape), float)  # Initialize the collective agent's weights
    weights = []
    for i in range(len(collective)): # i is the number of the agent
        agentWeights = np.zeros((len(data[0]),*collective[0].shape), float)  # Initialize the agent's weights
        
        for j in range(len(data[0])): # j is the time point
            # For each agent, multiply its weights by the data point
            for k in range(len(data[0][j])): # k is the number of the ground
                agentWeights[j,:,:,k,:] = collective[i][:,:,k,:]*data[0][j][k]
            for l in range(len(data[0][j])): # l is the index of each preceeding element inside the mask
                agentWeights[j,:,:,l,:] = collective[i][:,:,l,:]*data[1][j][l]  # Multiply the agent's weights by the data point
            colAgent[j,:,:,:,:] += agentWeights[j,:,:,:,:]
        weights.append(agentWeights)  # Append the agent's weights to the weights list

    weights.append(colAgent)
    # datapoints x options x options x grounds x 2

    options = collective[0].shape[0]  # Get the number of options from the shape of the weights
    # print('options', options)
    cdec = []
    for x in range(len(collective)+1):
        # print('processing agent', x+1, 'of', len(collective)+1)
        dec = []
        weight_system = weights[x]
        # print(weight_system.shape)
        for y in range(len(data[0])):
            # print('processing data point', y+1, 'of', len(data[0]))
            de = np.ones(options) # Initialize a value array with 1s for each option, 1 means permitted and 0 means not permitted
            for option1 in range(options):
                for option2 in range(option1,options):
                    if option1 != option2:
                        # print('comparing', option1, 'and', option2)
                        # print(de)
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

                        # print(jwo1, jwo2)
                        # Using single proportion because I was getting a lot of dillemma cases.
                        # v1 = np.sign(jwo1-jwo2)
                        # v2 = np.sign(jwo2-jwo1)
                        # print('v1', v1, 'v2', v2)

                        # Still a lot of dillemma cases. due to comparitivism.

                        # if v1 == -1: de[option1] = 0
                        if v1 == -1 and v2 != -1: de[option1] = 0
                        # if v2 == -1: de[option2] = 0 

            dec.append(de)

        # print('decision', len(dec), dec)
        cdec.append(dec)

    return (cdec, weights)


def evaluate(experiment: list, data: tuple) -> int:
    """we evaluate the experiment by comparing the decisions of the agents with the reference data.
    We understand [0,0] and [1,1] as hold and [0,1] and [1,0] as buy and sell.
    Remember that from the theory out 0 means forbidden and 1 means permitted."""
    # if the stock goes up, the decision should be buy or hold
    # if the stock goes down, the decision should be sell or hold

    output = experiment[0]  # Get the output of the experiment
    reference = experiment[1]  # Get the reference data of the experiment
    averages = data[0][:,0]  # Get the averages of the data
    RaR = data[0][:,1]  # Get the averages of the data

    # print(averages[:,0].shape)
    # print(output)
    delta = int(0)

    agents = len(output)  # The last element is the collective agent
    # print('agents', agents)

    evaluations = np.zeros((agents,2), int)  # Initialize the evaluations array

    for i in range(len(averages)-1):
        # print('average', averages[i])
        # print('output', output[0][i], output[1][i])
        delta = data[1][i+1][-1] - data[1][i][-1] 
        # print('delta', delta)
        for j in range(agents):
            # TODO is this logic correct? if current value < next value then we should buy now, if next value < current value than we should sell.
            if delta > 0:
                # If the stock goes up, the decision should be buy or hold 
                if output[j][i][0] == 1:
                    evaluations[j][0] += 1
                if output[j][i][1] == 1 and output[j][i][0] == 0:
                    evaluations[j][1] += 1
            elif delta < 0:
                # If the stock goes down, the decision should be sell or hold
                if output[j][i][1] == 1:
                    evaluations[j][0] += 1
                if output[j][i][0] == 1 and output[j][i][1] == 0:
                    evaluations[j][1] += 1
            else:
                # If the stock does not change, the decision should be hold
                if output[j][i][0] == output[j][i][1]:
                    evaluations[j][0] += 1
                else:
                    evaluations[j][1] += 1

    # print('evaluations', evaluations)
    # Return the evaluations
    return evaluations


def plot(evaluation: tuple, experiment: tuple, data: tuple) -> int:
    """This function plots a bar chart of a list of vectors of numbers."""
    
    ys = []
    xs = []
    for i in range(len(evaluation)-1):
        ys.append(evaluation[i][0])  
        ys.append(evaluation[i][1])  
        xs.append(f'Agent {i+1} correct')  
        xs.append(f'Agent {i+1} wrong')  

    ys.append(evaluation[-1][0])  
    ys.append(evaluation[-1][1])  
    xs.append('Collective Agent correct')  
    xs.append('Collective Agent wrong')  

    plt.figure(figsize=(len(ys), len(xs)))
    plt.bar(xs, ys)  # Create a bar chart with the data and names
    plt.title("Number of correct and wrong decisions per agent")
    plt.xlabel("Agents")
    plt.ylabel("number of occurances")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return 1

def main() -> int:
    """Main function to handle command line arguments."""

    mSize = 3  # Default mask size
    popSize = 5  # Default population size
    gs = 2  # Default number of grounds
    os = 2  # Default number of options

    data = get_data(maskSize = mSize, start_date = '2021-07-01', end_date = '2025-07-12')
    # print('data', data[0], data[1])
    # return 1
    population = generate_agents(populationSize = popSize, grounds = gs, options = os, maskSize = mSize)
    experiment = simulate_experiment(population, data)
    evaluation = evaluate(experiment, data)
    graph = plot(evaluation, experiment, data)

    return 1


if __name__ == '__main__':

    sys.exit(main())  
