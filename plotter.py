"""
Author: Ricardo Grande Cros
Created: 04/01/2022

This class contains useful methods for plotting TSP graphs
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio


def plot_network(cities, network, name='diagram.png'):
    """
    Simple plot
    todo: WORK IN PROGRESS
    :param cities:
    :param network:
    :param name:
    :return:
    """
    plt.xlim([0, 2])
    plt.ylim([0, 2])
    plt.scatter(cities[:, 0], cities[:, 1], s=2)
    plt.scatter(network[:, 0], network[:, 1], s=2)
    plt.savefig(f"./tmp/{name}", bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()


def plot_network2(cities, neurons, name='diagram.png', ax=None):
    """Plot a graphical representation of the problem"""
    mpl.rcParams['agg.path.chunksize'] = 10000

    if not ax:
        fig = plt.figure(figsize=(5, 5), frameon=False)
        axis = fig.add_axes([0, 0, 1, 1])

        axis.set_aspect('equal', adjustable='datalim')
        plt.axis('off')

        axis.scatter(cities[:, 0], cities[:, 1], color='red', s=2)
        axis.plot(neurons[:, 0], neurons[:, 1], 'r.', ls='-', color='#0063ba', markersize=2)

        plt.savefig(f"./tmp/{name}", bbox_inches='tight', pad_inches=0, dpi=200)
        plt.show()
        plt.close()

    else:
        ax.scatter(cities[:, 0], cities[:, 1], color='red', s=2)
        ax.plot(neurons[:, 0], neurons[:, 1], 'r.', ls='-', color='#0063ba', markersize=2)
        return ax


def plot_route(cities, route, name='diagram.png', ax=None):
    """Plot a graphical representation of the route obtained"""
    mpl.rcParams['agg.path.chunksize'] = 10000

    if not ax:
        fig = plt.figure(figsize=(5, 5), frameon=False)
        axis = fig.add_axes([0, 0, 1, 1])

        axis.set_aspect('equal', adjustable='datalim')
        plt.axis('off')

        axis.scatter(cities[:, 0], cities[:, 1], color='red', s=4)
        route = cities.reindex(route)
        route.loc[route.shape[0]] = route.iloc[0]
        axis.plot(route[:, 0], route[:, 1], color='purple', linewidth=1)

        plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()

    else:
        ax.scatter(cities[:, 0], cities[:, 1], color='red', s=4)
        route = cities.reindex(route)
        route.loc[route.shape[0]] = route.iloc[0]
        ax.plot(route[:, 0], route[:, 1], color='purple', linewidth=1)
        return ax


def generate_gif(filename_list):
    """
    Creates a gif animation from a list of images
    :param filename_list: list of image filenames
    :return: saves a gif movie in root path
    """
    images = []
    for filename in filename_list:
        images.append(imageio.imread(f'./tmp/{filename}'))
    imageio.mimsave("./movie.gif", images)
