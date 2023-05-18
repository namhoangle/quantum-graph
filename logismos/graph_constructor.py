from random import random

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import math


# class graph():


def create_ordering_array(n_slice_y, n_col_x, n_height_z):
    """
    Create ordering array for a 3D matrix of size n_slice_y x n_col_x x n_height_z.

    :param n_col_x:
    :param n_slice_y:
    :param n_height_z:
    :return:
    """
    flat_array = np.arange(1, n_slice_y * n_height_z * n_col_x + 1)
    # cost_array = flat_array.reshape(n_slice_y, n_col_x, n_height_z).transpose(0, 2, 1)

    order_array = flat_array.reshape(n_slice_y, n_col_x, n_height_z)

    return order_array


def assign_random_minus1_to_cost_matrix(cost):
    """
    Assigns a random element in each slice of cost to -1.

    Parameters:
        cost (ndarray): A 3-dimensional NumPy array of shape (nslice, nnode, ncol).

    Returns:
        ndarray: The updated cost array with one random element in each slice set to -1.
    """
    for i in range(cost.shape[0]): # iterate over slices
        for j in range(cost.shape[2]): # iterate over columns
            idx = random.randint(0, cost.shape[1]-1) # pick a random index
            cost[i, idx, j] = -1
    return cost


def plot_graph(cost_array):
    """
    Plot the order array as an image.

    :param cost_array:

    :return:
    """
    dim_slice, dim_node, dim_col = cost_array.shape

    # plot cost[0] as a 2D image
    nrows = math.ceil(dim_slice/4)
    figs, axs = plt.subplots(nrows=nrows, ncols=dim_slice, figsize=(dim_slice * 10, nrows * 10))

    count_ax=0
    for i in range(dim_slice):
        axs.flatten()[count_ax].imshow(cost_array[i], cmap=plt.cm.gray_r, interpolation='nearest')
        # show value in each cell of the matrix
        for x in range(cost_array[i].shape[0]):
            for y in range(cost_array[i].shape[1]):
                axs.flatten()[count_ax].text(y, x, '%.0f' % cost_array[i][x, y],
                                   horizontalalignment='center',
                                   verticalalignment='center',
                                   color='red',
                                   fontsize=30)
        axs.flatten()[count_ax].set_xticks([])
        axs.flatten()[count_ax].set_yticks([])
        count_ax += 1
        # ax.xlabel('columns')
        # ax.ylabel('')

    plt.show()


def build_graph_3d(n_height_z: int = 4,
                   n_width_x: int = 4,
                   n_depth_y: int = 1,
                   cost_matrix: np.ndarray or list = None):
    """

    :param n_height_z:
    :param n_width_x:
    :param n_depth_y:
    :param cost_matrix:

    :return:
    """
    # create zeroes matrix of size n_height x n_width x n_depth, and create a graph with n_height x n_width x n_depth nodes
    matrix = np.zeros((n_height_z, n_width_x, n_depth_y))

    # assign cost_matrix to matrix
    # for i in range(n_height):
    #     for j in range(n_width):
    #         for k in range(n_depth):
    #             matrix[i][j][k] = cost_matrix[i][j][k]

    matrix = cost_matrix


    # graph = nx.from_numpy_matrix(matrix)



    return matrix


# main function to run the program
if __name__ == '__main__':
    cost = create_ordering_array(4, 6, 5)
    print(cost)

    plot_graph(cost)


