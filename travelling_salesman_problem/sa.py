from simulated_annealing.travelling_salesman_problem.data import Data
import matplotlib.pyplot as plt
import numpy as np
import copy
import time
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
                 temp_initial,
                 temp_reduction_rate):

        self._cost_function = cost_function
        self._max_iteration = max_iteration
        self._max_sub_iteration = max_sub_iteration
        self._temp_initial = temp_initial
        self._temp_reduction_rate = temp_reduction_rate
        self._solution = None
        self._solution_new = None
        self._best_cost = []
        self._model_data = Data.load()

    @staticmethod
    def _roulette_wheel_selection(probs):

        number = np.random.random()
        probs_cumsum = np.cumsum(probs)

        return int(np.argwhere(number <= probs_cumsum)[0][0])

    def _initialize_evaluation_solution(self):

        solution = self._Individual()

        solution.position = np.random.permutation(len(self._model_data["location_x"]))\
            .reshape(1, len(self._model_data["location_x"]))

        solution.cost = self._cost_function(solution.position)

        return solution

    @staticmethod
    def _swap(position, min_index, max_index):

        out = copy.deepcopy(position)

        out[:, [min_index, max_index]] = position[:, [max_index, min_index]]

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

    @staticmethod
    def _reversion(position, min_index, max_index):

        out = np.concatenate((position[:, :min_index],
                              np.flip(position[:, min_index: max_index + 1]),
                              position[:, max_index + 1:]), axis=1)

        return out

    def _create_neighbor(self):

        solution_new = self._Individual()

        indices = [int(i) for i in np.random.choice(range(int(self._solution.position.shape[1])), 2, replace=False)]
        min_, max_ = min(indices), max(indices)

        method = self._roulette_wheel_selection([0.3, 0.3, 0.4])

        if method == 0:

            solution_new.position = self._swap(self._solution.position, min_, max_)

        elif method == 1:

            solution_new.position = self._insertion(self._solution.position, min_, max_)

        elif method == 2:

            solution_new.position = self._reversion(self._solution.position, min_, max_)

        solution_new.cost = self._cost_function(solution_new.position)

        return solution_new


    def run(self):

        tic = time.time()

        self._solution = self._initialize_evaluation_solution()

        for iter_main in range(self._max_iteration):

            for iter_sub in range(self._max_sub_iteration):

                self._solution_new = self._create_neighbor()

                if self._solution_new.cost < self._solution.cost:

                    self._solution = copy.deepcopy(self._solution_new)

                else:

                    delta = (self._solution_new.cost - self._solution.cost) / self._solution.cost

                    prob = np.exp(-1 * delta / self._temp_initial)

                    if np.random.rand() < prob:

                        self._solution = copy.deepcopy(self._solution_new)

            self._temp_initial *= self._temp_reduction_rate
            self._best_cost.append(self._solution.cost)

        toc = time.time()


        os.makedirs("./figures", exist_ok=True)

        location_x = self._model_data["location_x"]
        location_y = self._model_data["location_y"]

        tour = self._solution.position


        plt.figure(dpi=300, figsize=(10, 6))
        plt.plot(range(self._max_iteration), self._best_cost)
        plt.xlabel("Number of Iteration")
        plt.ylabel("Best Cost")
        plt.title("Travelling Salesman Problem Using Simulated Annealing", fontweight="bold")
        plt.savefig("./figures/cost_function_sa.png")

        plt.figure(dpi=300, figsize=(10, 6))

        plt.scatter(location_x, location_y, marker="o", c="blue")

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

        plt.savefig("./figures/tour.png")

        return self._solution, toc - tic
