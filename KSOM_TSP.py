"""
Author: Ricardo Grande Cros
Created: 05/01/2022

A TSP solver approach using Kohonen Self Organized Maps
"""
import math

import numpy as np

import plotter
from tools import *
from plotter import *


class KohonenTSP:
    def __init__(self, directory, filename, iterations, neighbourhood_size, initial_learning_rate):
        self.path = f'./{directory}/{filename}'
        self.directory = directory
        self.filename = filename

        # initialization before parsing
        self.problem_name = ""
        self.dimension = 0
        self.cities = []
        self.cities_normalized = []

        self.parse_tsp(self.path)

        # KSOM initialization (neighbourhood_size+2 to start with smallest possible network
        self.KSOM = np.random.uniform(low=0.45, high=0.55, size=(neighbourhood_size + 2, 2))
        self.winners = np.zeros(len(self.KSOM))

        self.iterations = iterations
        self.neighbourhood_size = neighbourhood_size
        self.initial_learning_rate = initial_learning_rate

        self.image_names = []

    def compute_distances(self, value):
        """
        Returns an array with the distances from a point to all KSOM's neurons
        :param value: a point (x,y)
        :return:
        """
        distances = []
        # distances = np.sqrt(np.sum((np.square(value - self.KSOM))))
        for s0, s1 in self.KSOM:
            dist_ = math.sqrt((value[0] - s0) ** 2 + (value[1] - s1) ** 2)  # Edit this line to [0]s and [1]s
            # print(dist)
            distances.append(dist_)  # Save data to list
        return np.array(distances, dtype=np.double)

    def compute_decay(self, value, iteration, variable_name=None):
        """
        Evaluates a given value with the exponential decay expression
        :param value: initial value to input in equation
        :param iteration: current iteration
        :param variable_name: learning_rate to differentiate the expression
        :return:
        """
        return value * np.exp(-iteration / (self.iterations*2))

    def update_weights(self, city, winner, learning_rate):
        # update weights of winner
        self.KSOM[winner] += learning_rate * (city - self.KSOM[winner])
        # update weights of neighbours
        for i in range(1, self.neighbourhood_size + 1):
            self.KSOM[(winner + i) % len(self.KSOM)] += \
                (learning_rate / (i+1)) * (city - self.KSOM[(winner + i) % len(self.KSOM)])
            self.KSOM[(winner - i) % len(self.KSOM)] += \
                (learning_rate / (i+1)) * (city - self.KSOM[(winner - i) % len(self.KSOM)])

    def train_ksom(self):
        """
        Contains KSOM-TSP training algorithm
        :return:
        """
        # initialize learning rate
        lr = self.initial_learning_rate
        # execute for n iterations
        for cycle in range(self.iterations):
            for i in range(self.dimension):
                # shuffle cities to ensure randomization when sampling
                np.random.shuffle(self.cities_normalized)
                # sample a random city
                city = self.cities_normalized[i]
                # obtain winner for the city sampled
                winner = np.argmin(self.compute_distances(city))
                self.winners[winner] += 1
                # update weights
                self.update_weights(city, winner, lr)
                # update learning rate
                lr = self.compute_decay(self.initial_learning_rate, cycle, 'learning_rate')

            # end of cycle
            # update winners matrix
            if len(self.KSOM) < self.dimension:
                # do mitosis only if size of network is less than number of cities
                big_winners = np.where(self.winners > 1)[0]
                self.KSOM = np.insert(self.KSOM, big_winners, self.KSOM[big_winners], axis=0)
                self.winners = np.insert(self.winners, big_winners, 1)
            # delete neurons that haven't won
            losers = np.where(self.winners == 0)[0]
            self.KSOM = np.delete(self.KSOM, losers, axis=0)
            self.winners = np.delete(self.winners, losers)
            # reset winners
            self.winners = np.zeros(len(self.KSOM))

            # plot every 10 iterations to make an animation
            image_name = f"image{cycle}_{i}.png"
            self.image_names.append(image_name)
            plot_neighbours(self.cities_normalized, self.KSOM, self.problem_name, lr, cycle, self.iterations, image_name)

        # print("POST: ", self.KSOM)
        print("Training completed. {} cycles executed".format(self.iterations))
        generate_gif(self.image_names)

    def parse_tsp(self, path):
        """
        Method for parsing TSP files into array. It also stores file name and
        the distance equation to use, as well as the problem's dimension.
        :param path: relative path to file
        :return:
        """
        content = read_file(path)
        # save problem name
        self.problem_name = [attribute.partition('NAME:')[2] for attribute in content if 'NAME:' in attribute][
            0].strip()

        self.dimension = int([line.partition('DIMENSION:')[2] for line in content if 'DIMENSION: ' in line][0])
        index_for_search = [index for index, line in enumerate(content) if 'NODE_COORD_SECTION' in line][0] + 1
        cities_data = content[index_for_search:index_for_search + self.dimension]
        self.cities = np.zeros((self.dimension, 2))

        for city in cities_data:
            index, xcoord, ycoord = map(float, list(filter(None, city.split(' '))))
            self.cities[int(index) - 1] = [xcoord, ycoord]

        self.cities_normalized = normalize_data(self.cities)


# ===================================================
# testing area
tsp = KohonenTSP("ALL_tsp", "att48.tsp", 500, 1, 0.05)
tsp.train_ksom()
