from simulated_annealing.travelling_salesman_problem.data import Data



def cost_function(x):

    distances = Data.load()["distances"]

    cost = 0
    for i in range(int(x.shape[1]) - 1):

        cost += distances[x[0, i], x[0, i + 1]]

    cost += distances[x[0, -1], x[0, 0]]

    return cost
