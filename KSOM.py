"""
Author: Ricardo Grande Cros
Created: 03/01/2022

Kohonen Self Organized Map

todo: redo file to work as generic KSOM
"""

import math

import numpy as np
from tools import read_file
from plotter import plot_network, plot_network2, generate_gif


class KSOM:

    def __init__(self, directory, filename, distance_equation, num_iterations, initial_learning_rate):
        """
        Constructor that initializes main SOM properties
        :param directory:
        :param filename:
        :param num_iterations:
        :param initial_learning_rate:
        """
        # save problem's path
        self.path = f'./{directory}/{filename}'
        self.name = ""
        # initialize cities array, it will be later modified @ parse_tsp
        self.cities = np.ndarray((1,))
        self.dimension = 0
        self.distance_equation = distance_equation
        # stores training iterations
        self.iterations = num_iterations

        self.parse_tsp(self.path)

        # stores a normalized version of cities array
        self.cities_normalized = self.normalize(self.cities)

        # initialize KSOM with 2*n prototypes
        # self.KSOM = np.random.rand(self.dimension, 2)  # 2,m dimension for 2D TSP
        self.KSOM = np.random.uniform(low=0.45, high=0.55, size=(self.dimension, 2))

        # Based on Zhang W., W. Jianwu, A deterministic self-organizing map approach and its application
        # on satellite data based cloud type classification, initial radius can be calculated as min(rows, columns)/2
        # self.initial_radius = min(self.dimension, 2) / 2
        self.initial_radius = 0.05
        self.initial_learning_rate = initial_learning_rate

        self.image_names = []

    def compute_distances(self, value):
        """
        Returns an array with the distances from a point to all KSOM's neurons
        :param value: a point (x,y)
        :return:
        """
        distances = []
        if self.distance_equation == 'euclidean':
            # distances = np.sqrt(np.sum((np.square(value - self.KSOM))))
            for s0, s1 in self.KSOM:
                dist_ = math.sqrt((value[0] - s0) ** 2 + (value[1] - s1) ** 2)  # Edit this line to [0]s and [1]s
                # print(dist)
                distances.append(dist_)  # Save data to list
        if self.distance_equation == 'scalar_product':
            distances = np.sum(self.KSOM * value, axis=1)
        return np.array(distances, dtype=np.double)

    def compute_neighbourhood(self, iteration, winner):
        """
        Returns a vtor with the gaussian neighbourhood expression
        :param iteration: current iteration
        :param winner: current iteration's winner cell
        :return:
        """
        radius = self.compute_decay(self.initial_radius, iteration)
        distances = self.compute_distances(self.KSOM[winner])
        return np.exp(- (distances * distances) / (2 * radius * radius))

    def get_neighborhood(self, center, radix, domain):
        """Get the range gaussian of given radix around a center index."""

        # Impose an upper bound on the radix to prevent NaN and blocks
        if radix < 1:
            radix = 1

        # Compute the circular network distance to the center
        deltas = np.absolute(center - np.arange(domain))
        distances = np.minimum(deltas, domain - deltas)

        # Compute Gaussian distribution around the given center
        return np.exp(-(distances * distances) / (2 * (radix * radix)))

    def compute_decay(self, value, iteration, variable_name=None):
        """
        Evaluates a given value with the exponential decay expression
        :param value: initial value to input in equation
        :param iteration: current iteration
        :param variable_name: learning_rate to differentiate the expression
        :return:
        """
        return value * np.exp(iteration / self.iterations) if variable_name == 'learning_rate' \
            else value * np.exp(-iteration / self.iterations*10)

    def normalize(self, vector):
        """
        Returns a normalized vector
        :param vector: vector to normalize
        :return: normalized vector
        """
        vector[:, 0] = (vector[:, 0] - vector[:, 0].min()) / (vector[:, 0].max() - vector[:, 0].min())
        vector[:, 1] = (vector[:, 1] - vector[:, 1].min()) / (vector[:, 1].max() - vector[:, 1].min())

        return vector

    def train_KSOM(self):
        """
        Contains KSOM training algorithm
        :return:
        """
        # initialize learning rate
        lr = self.initial_learning_rate
        # execute for n iterations
        for i in range(self.iterations):
            # sample a random city
            input_index = np.random.choice(tsp.cities.shape[0], size=1, replace=False)[0]
            city = self.cities_normalized[input_index]
            # obtain winner for the city sampled
            winner = np.argmin(self.compute_distances(city))
            # update weight vectors based on neighbourhood
            # w(t+1) = w(t) + lr(t)*g(t,x)*(x - w(t))
            #print("-------- PRE --------")
            #print("pre: ", self.KSOM[0])
            #print("learning_rate: ", lr, " ---- neighbourhood: ", self.compute_neighbourhood(i, winner)[0], " ---- subtraction: ", (city - self.KSOM)[0])
            #print("increment: ", (lr * self.compute_neighbourhood(i, winner)[:, np.newaxis] * (city - self.KSOM)))
            self.KSOM += (lr * self.compute_neighbourhood(i, winner)[:, np.newaxis] * (city - self.KSOM))

            #print("result: ", self.KSOM[0])
            #self.KSOM += (lr/self.compute_distances(self.KSOM[winner]))[:, np.newaxis] * (city - self.KSOM)

            # Solución sin vecindario
            #incremento = lr * (city - self.KSOM[winner])
            #print("incremento: ", incremento, " --- lr: ", lr, " --- city[0]: ", city, " --- ksom: ", self.KSOM[winner])
            #self.KSOM[winner][0] = self.KSOM[winner][0] + incremento[0]
            #self.KSOM[winner][1] = self.KSOM[winner][1] + incremento[1]

            # update learning rate
            lr = self.compute_decay(self.initial_learning_rate, i, 'learning_rate')

            if not i % 1000:
                # progress print
                print("{} iterations completed".format(i))
            if not i % 100:
                # plot every 10 iterations to make an animation
                image_name = f"image_{i}.png"
                self.image_names.append(image_name)
                plot_network(self.cities_normalized, self.KSOM, image_name)

        # print("POST: ", self.KSOM)
        print("Training completed. {} iterations executed".format(self.iterations))
        generate_gif(self.image_names)
        print(self.KSOM)

    def parse_tsp(self, path):
        """
        Method for parsing TSP files into array. It also stores file name and
        the distance equation to use, as well as the problem's dimension.
        :param path: relative path to file
        :return:
        """
        content = read_file(path)
        # save problem name
        self.name = [attribute.partition('NAME:')[2] for attribute in content if 'NAME:' in attribute][0].strip()

        self.dimension = int([line.partition('DIMENSION:')[2] for line in content if 'DIMENSION: ' in line][0])
        index_for_search = [index for index, line in enumerate(content) if 'NODE_COORD_SECTION' in line][0] + 1
        cities_data = content[index_for_search:index_for_search + self.dimension]
        self.cities = np.zeros((self.dimension, 2))

        for city in cities_data:
            index, xcoord, ycoord = map(float, list(filter(None, city.split(' '))))
            self.cities[int(index) - 1] = [xcoord, ycoord]


# ===================================================
# testing area
tsp = KSOM("ALL_tsp", "burma14.tsp", "euclidean", 10000, 0.01)
tsp.train_KSOM()
