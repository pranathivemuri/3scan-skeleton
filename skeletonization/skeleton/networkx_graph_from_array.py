import itertools

import numpy as np
import networkx as nx
from scipy import ndimage

import skeleton.image_tools as image_tools

"""
program to look up adjacent elements and calculate degree
this dictionary can be used for graph creation
since networkx graph based on looking up the array and the
adjacent coordinates takes long time. create a dict
using dict_of_indices_and_adjacent_coordinates.
Following are the 27 position vectors of 3 x 3 x 3 second ordered neighborhood of a voxel
at origin (0, 0, 0)
(-1 -1 -1) (-1 0 -1) (-1 1 -1)
(-1 -1 0)  (-1 0 0)  (-1 1 0)
(-1 -1 1)  (-1 0 1)  (-1 1 1)
(0 -1 -1) (0 0 -1) (0 1 -1)
(0 -1 0)  (0 0 0)  (0 1 0)
(0 -1 1)  (0 0 1)  (0 1 1)
(1 -1 -1) (1 0 -1) (1 1 -1)
(1 -1 0)  (1 0 0)  (1 1 0)
(1 -1 1)  (1 0 1)  (1 1 1)
"""
TEMPLATE = np.array([[[33554432, 16777216, 8388608], [4194304, 2097152, 1048576], [524288, 262144, 131072]],
                    [[65536, 32768, 16384], [8192, 0, 4096], [2048, 1024, 512]],
                    [[256, 128, 64], [32, 16, 8], [4, 2, 1]]], dtype=np.uint64)


# permutations of (-1, 0, 1) in three/two dimensional tuple format
# representing 8 and 26 increments around a pixel at origin (0, 0, 0)
# 2nd ordered neighborhood around a voxel/pixel
POSITION_VECTORS = list(itertools.product((-1, 0, 1), repeat=3))
POSITION_VECTORS.remove((0, 0, 0))


def _get_position_vectors(config_number: int):
    """
    Return a list of tuples of position vectors of the bitmask
    for a given config_number
    Parameters
    ----------
    config_number : int64
        integer less than 2 ** 26
    Returns
    -------
    list
        a list of position vectors of a non zero voxel/pixel
        if it is a zero voxel, an empty tuple is returned, else
        the position vector of non zero voxel is returned
    Notes
    ------
    As in the beginning of the program, there are position vectors
    around a voxel at origin (0, 0, 0) which are returned by this function.
    config_number is a decimal number representation of 26 binary numbers
    around a voxel at the origin in a second ordered neighborhood
    """
    neighbor_values = [(config_number >> digit) & 0x01 for digit in range(26)]
    return [neighbor_val * position_vec for neighbor_val, position_vec in zip(neighbor_values, POSITION_VECTORS)]


def set_adjacency_list(coordinate_bitmask_lists: list, arr_lower_limits: tuple, arr_upper_limits):
    """
    Return dict
    Parameters
    ----------
    coordinate_bitmask_lists : list
        list of two element lists, [[nonzero_coordinate, bitmask_config_number], [..]]
    arr_lower_limits: tuple
        tuple representing the lower limits of the array in which the graph is located
    arr_upper_limits: tuple
        tuple representing the lower limits of the array in which the graph is located
    Returns
    -------
    dict_of_indices_and_adjacent_coordinates: Dictionary
        key is the nonzero coordinate
        is all the position of nonzero coordinates around it
        in it's second order neighborhood
    """
    arr_lower_limits, arr_upper_limits
    dict_of_indices_and_adjacent_coordinates = {}
    # list of unique nonzero tuples
    for nonzero, config_number in coordinate_bitmask_lists:
        adjacent_coordinate_list = [tuple(np.add(nonzero, position_vector))
                                    for position_vector in _get_position_vectors(config_number)
                                    if position_vector != () and
                                    tuple(np.add(nonzero, position_vector)) >= arr_lower_limits and
                                    tuple(np.add(nonzero, position_vector)) < arr_upper_limits]
        dict_of_indices_and_adjacent_coordinates[tuple(nonzero)] = adjacent_coordinate_list
    return dict_of_indices_and_adjacent_coordinates


def get_coord_bitmasks(arr: np.array,
                       offset: list= [0, 0, 0]):
    """
    Return list of two element lists, nonzero_coordinate coordinate and second element is bitmask_config_number
    All coordinates will be translated by the offset parameter, which defaults to no translation
    Parameters
    ----------
    arr : np.array
        3D boolean array
    offset: list of 3 values, can be positive or negative
        offset in voxel coordinates, this value is added to the coordinates of nonzero elements in `arr`,
        default nothing is added
    Returns
    -------
    coordinate_bitmask_lists : list
        list of two element lists, [[nonzero_coordinate, bitmask_config_number], [..]]
    """
    # convert the binary array to a configuration number array of same size by convolving with template
    if arr.sum() == 0 or arr.sum() == arr.size:
        return []
    result = ndimage.convolve(np.uint64(arr), TEMPLATE, mode='constant')
    nonzero_coordinates = image_tools.list_of_tuples_of_val(arr, 1)
    coordinate_bitmask_lists = []
    for nonzero in nonzero_coordinates:
        coord_bitmask = [[int(posn + offset[i]) for i, posn in enumerate(nonzero)], int(result[nonzero])]
        coordinate_bitmask_lists.append(coord_bitmask)
    return coordinate_bitmask_lists


def _get_cliques_of_size(networkx_graph, clique_size):
    """
    Return cliques of size "clique_size" in networkx_graph
    Parameters
    ----------
    networkx_graph : Networkx graph
        graph to obtain cliques from
    clique_size : int
        size = number of edges in the clique forming cycle
    Returns
    -------
        list
        list of edges forming 3 vertex cliques
    """
    cliques = nx.find_cliques_recursive(networkx_graph)
    # all the nodes/vertices of 3 cliques
    return [clique for clique in cliques if len(clique) == clique_size]


def _reduce_clique(clique_edges, combination_edges, mth_clique, mth_clique_edge_length_list):
    """
    a) the edge with maximum edge length in case of a right angled clique (1, 1, sqrt(2)) or (1, 2, sqrt())
    """
    for nth_edge_in_mth_clique, edge_length in enumerate(mth_clique_edge_length_list):
        if edge_length == np.max(mth_clique_edge_length_list):
            clique_edges.append(combination_edges[mth_clique][nth_edge_in_mth_clique])


def _remove_clique_edges(networkx_graph):
    """
    Return 3 vertex clique removed networkx graph changed in place
    Parameters
    ----------
    networkx_graph : Networkx graph
        graph to remove cliques from
    Returns
    -------
    networkx_graph: Networkx graph changed in place
        graph with 3 vertex clique edges removed
    Notes
    ------
    Returns networkx graph changed in place after removing 3 vertex cliques
    Removes the longest edge in a 3 vertex cliques and
    clique forming edge in special case edges with equal
    lengths that form the 3 vertex clique.
    Doesn't deal with any other cliques.
    """
    three_vertex_cliques = _get_cliques_of_size(networkx_graph, clique_size=3)
    combination_edges = [list(itertools.combinations(clique, 2)) for clique in three_vertex_cliques]
    # clique_edge_lengths is a list of lists, where each list is the length of an edge in 3 vertex clique
    clique_edge_lengths = []
    # different combination of edges are in combination_edges and their corresponding lengths are in
    # clique_edge_lengths
    for combination_edge in combination_edges:
        clique_edge_lengths.append([np.sum((np.array(edge[0]) - np.array(edge[1])) ** 2)
                                    for edge in combination_edge])
    # clique edges to be removed are collected here in the for loop below
    clique_edges = []
    for mth_clique, mth_clique_edge_length_list in enumerate(clique_edge_lengths):
        _reduce_clique(clique_edges, combination_edges, mth_clique, mth_clique_edge_length_list)

    networkx_graph.remove_edges_from(clique_edges)


def get_networkx_graph_from_array(coordinate_bitmask_lists: list, arr_lower_limits: tuple, arr_upper_limits: tuple):
    """
    Return a networkx graph, Raise an assertion error if any new non zero coordinates on skeleton
    are introduced when the skeleton is converted to graph
    Parameters
    ----------
    coordinate_bitmask_lists : list
        list of two element lists, [[nonzero_coordinate, bitmask_config_number], [..]]
    arr_lower_limits: tuple
        tuple representing the lower limits of the array in which the graph is located
    arr_upper_limits: tuple
        tuple representing the lower limits of the array in which the graph is located
    Returns
    -------
    networkx_graph : Networkx graph
        graphical representation of the input array after clique removal
    Note
    ----
    arr_lower_limits and arr_upper_limits are used to contain the edges of the graph to be contained within a bounding box
    """
    networkx_graph = nx.from_dict_of_lists(set_adjacency_list(coordinate_bitmask_lists, arr_lower_limits, arr_upper_limits))
    _remove_clique_edges(networkx_graph)
    assert networkx_graph.number_of_nodes() == len(coordinate_bitmask_lists)
    return networkx_graph
