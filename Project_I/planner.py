import numpy as np
from typing import List, Tuple, Optional
import heapq

def find_path(grid, start, end):
    """
    Finds the optimal path from the start to end position in a grid
    using the Chebyshev distance formula of max(|x1 - x2|, |y1 - y2|)
    to determine the next most optimal move at each step.

    Parameters:
    - grid: List of lists showcasing the grid environment.
        - 0 represents a walkable cell.
        - 1 represents an obstacle.
    - start (Tuple[int, int]): The (row, column) coordinates of the starting position.
    - end (Tuple[int, int]): The (row, column) coordinates of the goal position.
    
    Returns:
    - Optimal path to arrive at the goal in the least amount of steps possible.
      Note: This returns one of many optimal paths given that there are generally
            going to be multiple possibile optimal paths.
    """

    rows, cols = len(grid), len(grid[0])
    heap = [(abs(start[0] - end[0]) + abs(start[1] - end[1]), 0, 0, start)]
    parent = {start: None}
    visited = {start: 0}

    # Consider all 8 possible moves (up, down, left, right, and diagonals)
    directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0),  # Up, Down, Left, Right
                  (-1, -1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, 1)]  # Diagonal moves
    
    # Loop until there are no values left in the heap
    while heap:
        # Pull the next grid with the lowest f(n) cost
        _, g, _, (x, y) = heapq.heappop(heap)

        # If the goal node is reached, reconstruct the path taken
        if (x, y) == end:
            path = []
            while (x, y) is not None:
                path.append((x, y))
                if parent[(x, y)] is None:
                    break
                x, y = parent[(x, y)]
            return path[::-1]

        # Loop through all possible nearby moves and add the best, valid options 
        for dx, dy, weight in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
                new_g = g + 1
                # Calculate the f(n) from g(n) + h(n) using Chebyshev distance
                new_f = new_g + max(abs(nx - end[0]), abs(ny - end[1]))

                # If the node has not been visited or a new, shorter path has been found
                # Then visit the node and add it to the heap
                if (nx, ny) not in visited or new_g < visited[(nx, ny)]:
                    visited[(nx, ny)] = new_g
                    heapq.heappush(heap, (new_f, new_g, weight, (nx, ny)))
                    parent[(nx, ny)] = (x, y)

    # Return none since no optimal path was found (the goal could not be reached)
    return None

def plan_path(world: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[np.ndarray]:

    """
    Computes a path from the start position to the end position 
    using a certain planning algorithm (DFS is provided as an example).

    Parameters:
    - world (np.ndarray): A 2D numpy array representing the grid environment.
      - 0 represents a walkable cell.
      - 1 represents an obstacle.
    - start (Tuple[int, int]): The (row, column) coordinates of the starting position.
    - end (Tuple[int, int]): The (row, column) coordinates of the goal position.

    Returns:
    - np.ndarray: A 2D numpy array where each row is a (row, column) coordinate of the path.
      The path starts at 'start' and ends at 'end'. If no path is found, returns None.
    """
    # Ensure start and end positions are tuples of integers
    start = (int(start[0]), int(start[1]))
    end = (int(end[0]), int(end[1]))

    # Convert the numpy array to a list of lists
    world_list: List[List[int]] = world.tolist()

    # Perform path finding
    path = find_path(world_list, start, end)

    return np.array(path) if path else None
