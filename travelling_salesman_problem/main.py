from simulated_annealing.travelling_salesman_problem.data import Data
from simulated_annealing.travelling_salesman_problem.travelling_salesman_problem import cost_function
# from simulated_annealing.travelling_salesman_problem.sa import SA
from simulated_annealing.travelling_salesman_problem.sa_population_based import SA


def main():


    # model_data = Data(number_of_nodes=50)
    # model_data.create_and_save()
    # data = Data.load()

    cost_func = cost_function

    # solution_method = SA(cost_function=cost_func,
    #                      max_iteration=1000,
    #                      max_sub_iteration=10,
    #                      temp_initial=1,
    #                      temp_reduction_rate=0.99)
    #
    # solution_best, run_time = solution_method.run()
    #
    # return solution_best, run_time

    solution_method = SA(cost_function=cost_func,
                         max_iteration=50,
                         max_sub_iteration=10,
                         number_of_population=10,
                         number_of_neighbors=10,
                         temp_initial=1,
                         temp_reduction_rate=0.99)

    solution_best, run_time = solution_method.run()

    return solution_best, run_time


if __name__ == "__main__":

    solution, runtime = main()
