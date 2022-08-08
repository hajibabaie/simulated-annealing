from simulated_annealing.travelling_salesman_problem.data import Data
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import os


class SA:

    class _Individual:

        def __init__(self):

            self.position = None
            self.cost = None

    def __init__(self,
                 cost_function,
                 max_iteration,
                 max_sub_iteration,
                 number_of_population,
                 number_of_neighbors,
                 temp_initial,
                 temp_reduction_rate,
                 ):


        self._cost_function = cost_function
        self._max_iteration = max_iteration
        self._max_sub_iteration = max_sub_iteration
        self._number_of_population = number_of_population
        self._number_of_neighbors = number_of_neighbors
        self._temp_initial = temp_initial
        self._temp_reduction_rate = temp_reduction_rate
        self._population_main = None
        self._population_new = []
        self._best_cost = []
        self._model_data = Data.load()


    @staticmethod
    def _roulette_wheel_selection(probs):

        number = np.random.rand()

        probs_cumsum = np.cumsum(probs)

        return int(np.argwhere(number < probs_cumsum)[0][0])

    def _initialize_evaluate_population(self):

        out = [self._Individual() for _ in range(self._number_of_population)]

        for i in range(self._number_of_population):

            out[i].position = np.random.permutation(len(self._model_data["location_x"])).\
                reshape(1, len(self._model_data["location_x"]))

            out[i].cost = self._cost_function(out[i].position)

        return out

    @staticmethod
    def _sort(population):

        population_cost_argsort = [int(i) for i in np.argsort([population[j].cost for j in range(len(population))])]

        pop = [population[i] for i in population_cost_argsort]

        return pop

    @staticmethod
    def _swap(position, min_index, max_index):

        out = copy.deepcopy(position)

        out[0, [min_index, max_index]] = position[0, [max_index, min_index]]

        return out

    @staticmethod
    def _reversion(position, min_index, max_index):

        out = np.concatenate((position[:, :min_index],
                              np.flip(position[:, min_index: max_index + 1]),
                              position[:, max_index + 1:]), axis=1)

        return out

    def _insertion(self, position, min_index, max_index):

        method = self._roulette_wheel_selection([0.5, 0.5])

        if method == 0:

            out = np.concatenate((position[:, :min_index + 1],
                                  position[:, max_index: max_index + 1],
                                  position[:, min_index + 1: max_index],
                                  position[:, max_index + 1:]), axis=1)

            return out

        elif method == 1:

            out = np.concatenate((position[:, :min_index],
                                  position[:, min_index + 1: max_index + 1],
                                  position[:, min_index: min_index + 1],
                                  position[:, max_index + 1:]), axis=1)

            return out

    def _create_neighbor(self, index_for_population):

        new = self._Individual()

        position = self._population_main[index_for_population].position

        indices = [int(i) for i in np.random.choice(range(int(position.shape[1])), 2, replace=False)]
        min_, max_ = min(indices), max(indices)

        method = self._roulette_wheel_selection([0.3, 0.3, 0.4])

        if method == 0:

            new.position = self._swap(position, min_, max_)

        elif method == 1:

            new.position = self._insertion(position, min_, max_)

        elif method == 2:

            new.position = self._reversion(position, min_, max_)

        new.cost = self._cost_function(new.position)

        return new


    def run(self):

        tic = time.time()

        self._population_main = self._initialize_evaluate_population()

        self._population_main = self._sort(self._population_main)

        for iter_main in range(self._max_iteration):

            for iter_sub in range(self._max_sub_iteration):

                for p in range(self._number_of_population):

                    for q in range(self._number_of_neighbors):

                        self._population_new.append(self._create_neighbor(p))

                self._population_new = self._sort(self._population_new)
                self._population_new = self._population_new[:self._number_of_population]

                for i in range(self._number_of_population):

                    if self._population_new[i].cost < self._population_main[i].cost:

                        self._population_main[i] = copy.deepcopy(self._population_new[i])

                    else:

                        delta = (self._population_new[i].cost - self._population_main[i].cost) / \
                                self._population_main[i].cost

                        p = np.exp(-1 * delta / self._temp_initial)

                        if np.random.rand() < p:

                            self._population_main[i] = copy.deepcopy(self._population_new[i])

            self._temp_initial *= self._temp_reduction_rate
            self._best_cost.append(self._population_main[0].cost)


        toc = time.time()

        os.makedirs("./figures", exist_ok=True)

        location_x = self._model_data["location_x"]
        location_y = self._model_data["location_y"]

        tour = self._population_main[0].position

        plt.figure(dpi=300, figsize=(10, 6))
        plt.plot(range(self._max_iteration), self._best_cost)
        plt.xlabel("Number of Iteration")
        plt.ylabel("Best Cost")
        plt.title("Travelling Salesman Problem Using Population Based Simulated Annealing", fontweight="bold")
        plt.savefig("./figures/cost_function_sa_population_based.png")

        plt.figure(dpi=300, figsize=(10, 6))

        plt.scatter(location_x, location_y)
        for i in range(len(location_x)):

            plt.text(location_x[i], location_y[i], str(i))

        for i in range(len(location_x) - 1):

            if i == 0:

                plt.plot([location_x[tour[0, i]], location_x[tour[0, i + 1]]],
                         [location_y[tour[0, i]], location_y[tour[0, i + 1]]], color="green")

            else:

                plt.plot([location_x[tour[0, i]], location_x[tour[0, i + 1]]],
                         [location_y[tour[0, i]], location_y[tour[0, i + 1]]], color="black")

        plt.plot([location_x[tour[0, -1]], location_x[tour[0, 0]]],
                 [location_y[tour[0, -1]], location_y[tour[0, 0]]], color="red")

        plt.savefig("./figures/tour_sa_population_based.png")

        return self._population_main[0], toc - tic
