# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import List, Tuple, Set

import cirq
import numpy as np
import heapq


def get_indices_sorted_by_classical_time_approx(ham, msmt_ind=0, bfly_ind=1):
    """Calculates shortest classical path length of m -> j -> b in 1/H, and sorts indices by this"""

    weight_mat = 1 / abs(ham)
    paths_left = dijkstra(weight_mat, msmt_ind)
    paths_right = dijkstra(weight_mat, bfly_ind)
    paths_full = paths_left + paths_right

    indices_full = np.argsort(paths_full)
    paths_full = np.sort(paths_full)
    ind0 = list(indices_full).index(msmt_ind)
    indices_full[0], indices_full[ind0] = indices_full[ind0], indices_full[0]
    ind1 = list(indices_full).index(bfly_ind)
    indices_full[1], indices_full[ind1] = indices_full[ind1], indices_full[1]

    minpath = paths_full[0]
    num_spins_min = len([x for x in paths_full if abs(x - minpath) < 1e-6])
    return paths_full, indices_full, num_spins_min


def dijkstra(weight_matrix, start_node):
    """
    Compute the shortest path distances from a starting node to all other nodes in a graph
    represented by a weight matrix using Dijkstra's algorithm.

    Args:
        weight_matrix: A NumPy array representing the weight matrix of the graph.
        start_node: An integer representing the index of the starting node.

    Returns:
        A NumPy array of shortest path distances from the start node to all other nodes.
    """

    n_nodes = weight_matrix.shape[0]
    distances = np.full(n_nodes, np.inf)  # Initialize distances to infinity
    distances[start_node] = 0  # Distance from start node to itself is 0
    visited = np.zeros(n_nodes, dtype=bool)  # Keep track of visited nodes

    priority_queue = [(0, start_node)]  # (distance, node) tuples
    while priority_queue:
        dist, current_node = heapq.heappop(priority_queue)

        if visited[current_node]:
            continue
        visited[current_node] = True

        for neighbor_node in range(n_nodes):
            weight = weight_matrix[current_node, neighbor_node]
            if weight != np.inf:  # Check for existence of an edge
                new_dist = dist + weight
                if new_dist < distances[neighbor_node]:
                    distances[neighbor_node] = new_dist
                    heapq.heappush(priority_queue, (new_dist, neighbor_node))

    return distances


def find_hamiltonian_paths(grid: List[cirq.GridQubit]) -> List[List[Tuple[int, int]]]:
    """
    Finds all Hamiltonian paths on a finite grid.

    Args:
        grid_size: A tuple representing the (rows, columns) dimensions of the grid.

    Returns:
        A list of Hamiltonian paths. Each path is represented as a list of tuples,
        where each tuple represents the (row, column) coordinates of a cell in the path.
    """

    num_cells = len(grid)
    grid_tuples = [(q.row, q.col) for q in grid]

    def is_valid(row, col):
        return (row, col) in grid_tuples

    def find_paths(current_path, visited):
        if len(current_path) == num_cells:
            return [current_path]

        paths = []
        row, col = current_path[-1]
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_row, next_col = row + dr, col + dc
            if is_valid(next_row, next_col) and (next_row, next_col) not in visited:
                new_path = current_path + [(next_row, next_col)]
                new_visited = visited.union({(next_row, next_col)})
                paths.extend(find_paths(new_path, new_visited))
        return paths

    all_paths = []
    for q in grid:
        all_paths.extend(find_paths([(q.row, q.col)], {(q.row, q.col)}))

    all_paths_qubit = [[cirq.GridQubit(*pair) for pair in path] for path in all_paths]

    return all_paths_qubit
