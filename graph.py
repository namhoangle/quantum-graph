import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import math


# class graph():


def create_cost_array(nslice, nnode, ncol, plot: bool = False):
    flat_array = np.arange(1, nslice*nnode*ncol + 1)
    cost_array = flat_array.reshape(nslice, nnode, ncol).transpose(0, 2, 1)

    return cost_array


def plot_graph(cost_array):
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
                axs[count_ax].text(y, x, '%.0f' % cost_array[i][x, y],
                                   horizontalalignment='center',
                                   verticalalignment='center',
                                   color='red',
                                   fontsize=30)
        axs[count_ax].set_xticks([])
        axs[count_ax].set_yticks([])
        count_ax += 1
        # ax.xlabel('columns')
        # ax.ylabel('')


    plt.show()


def build_graph_3d(n_height: int,
                   n_width: int,
                   n_depth: int,
                   cost_matrix: np.ndarray):
    # create zeroes matrix of size n_height x n_width x n_depth, and create a graph with n_height x n_width x n_depth nodes
    matrix = np.zeros((n_height, n_width, n_depth))

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
    n = 4  # the size of the matrix

    cost_0 = [[ 1,  5,  9, 13],
              [ 2,  6, 10, 14],
              [ 3,  7, 11, 15],
              [ 4,  8, 12, 16]]
    cost_1 = [[17, 21, 25, 29],
                [18, 22, 26, 30],
                [19, 23, 27, 31],
                [20, 24, 28, 32]]
    cost_2 = [[33, 37, 41, 45],
                [34, 38, 42, 46],
                [35, 39, 43, 47],
                [36, 40, 44, 48]]
    cost_3 = [[49, 53, 57, 61],
                [50, 54, 58, 62],
                [51, 55, 59, 63],
                [52, 56, 60, 64]]

    cost = np.array([cost_0, cost_1, cost_2, cost_3])

    graph = build_graph_3d(n, n, n, cost)
    print(graph)

    # create the 3D plot
    dim_slice, dim_intracol, dim_intercol = graph.shape

    # plot cost[0] as a 2D image
    figs, axs = plt.subplots(nrows=math.ceil(dim_slice%8), ncols=8, figsize=(10, 3))

    count_ax=0
    for i in range(dim_slice):
        axs[count_ax].imshow(cost[i], cmap=plt.cm.gray_r, interpolation='nearest')
        # show value in each cell of the matrix
        for x in range(cost[i].shape[0]):
            for y in range(cost[i].shape[1]):
                axs[count_ax].text(y, x, '%.0f' % cost[0][x, y],
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='red')
        axs[count_ax].set_xticks([])
        axs[count_ax].set_yticks([])
        count_ax += 1
        # ax.xlabel('columns')
        # ax.ylabel('')


    plt.show()


