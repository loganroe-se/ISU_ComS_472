import numpy as np
from heapq import heappush, heappop
from typing import Optional, Tuple

# Scoring System - Weighting
#   Pursuer Weight - The higher the number, the more the algorithm favors staying away from pursuer
#   Pursued Weight - The higher the number, the more the algorithm favors aggressiveness
#   Escape Routes - The higher the number, the more the algorithm favors staying in areas with more options (i.e. no walls)
#   Wall Penalty - Set to avoid hitting edges
WEIGHT_PURSUER = 1.2
WEIGHT_PURSUED = 1.2
ESCAPE_ROUTES = 0.4
WALL_PENALTY = 1.0

# Define the manhattan distance at which to switch to defensive mode
DEFENSIVE_MANHATTAN_DIST = 3

# Define Available Directions
DIRECTIONS = [(0, 0),                                # Staying still
              (-1, 0), (1, 0), (0, -1), (0, 1),      # Up, down, left, right
              (-1, -1), (-1, 1), (1, -1), (1, 1)]    # Diagonal Moves



# Define a function to get manhattan distance
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Define an A* function to be the primary searching algorithm
def a_star(grid, start, end):
    rows, cols = grid.shape
    heap = [(0 + manhattan(start, end), 0, start)]
    parent = {}
    g_score = {start: 0}

    while heap:
        _, _, current = heappop(heap)

        # If the end has been reached, reconstruct the path
        if current == end:
            path = [current]
            while current in parent:
                current = parent[current]
                path.append(current)
            return path[::-1]
        
        # Loop through available directions
        for dx, dy in DIRECTIONS:
            nx, ny = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)
            # Ensure the move is valid
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
                new_g = g_score[current] + 1
                # If a new neighbor, or a better path, is found, update values
                if neighbor not in g_score or new_g < g_score[neighbor]:
                    parent[neighbor] = current
                    g_score[neighbor] = new_g
                    f_score = new_g + manhattan(neighbor, end)
                    heappush(heap, (f_score, new_g, neighbor))

    # Return none if no moves were found
    return None

# Define a function to find the prediction of the next position
def predict_next_pos(grid, start, target):
    path = a_star(grid, tuple(start), tuple(target))
    if path and len(path) > 1:
        return path[1]
    return tuple(start)


# Define the PlannerAgent class
class PlannerAgent:
    def __init__(self):
        pass

    # Define the plan_action function to determine the next best move
    def plan_action(self, world: np.ndarray, current: Tuple[int, int], 
                    pursued: Tuple[int, int], pursuer: Tuple[int, int]) -> Optional[np.ndarray]:        
        # Convert to numpy arrays for certain calculations
        current_np = np.array(current)
        pursued_np = np.array(pursued)
        pursuer_np = np.array(pursuer)

        # Predict next positions
        predicted_pursuer = predict_next_pos(world, pursuer_np, current_np)
        predicted_pursued = predict_next_pos(world, pursued_np, pursuer_np)

        # If the pursuer is close, use a weighted system to determine the next move
        if manhattan(current, pursuer) <= DEFENSIVE_MANHATTAN_DIST:
            best_move = np.array([0, 0])
            best_score = -float('inf')

            # Loop through all possible moves
            for d in DIRECTIONS:
                x, y = current_np + d
                # If the next move is valid, continue
                if 0 <= x < world.shape[0] and 0 <= y < world.shape[1] and world[x, y] == 0:
                    dist_from_pursuer = manhattan((x, y), predicted_pursuer)
                    dist_to_pursued = manhattan((x, y), predicted_pursued)

                    # Determine the number of escape routes (directions that are not an obstacle/wall)
                    escape_routes = sum(
                        1 for dir in DIRECTIONS
                        if 0 <= x + dir[0] < world.shape[0]
                        and 0 <= y + dir[1] < world.shape[1]
                        and world[x + dir[0], y + dir[1]] == 0
                    )

                    # Calculate the wall penalty (0 or 1)
                    wall_penalty = int(x in [0, world.shape[0] - 1] or y in [0, world.shape[1] - 1])

                    # Calculate the weighted score
                    score = (
                        WEIGHT_PURSUER * dist_from_pursuer -
                        WEIGHT_PURSUED * dist_to_pursued +
                        ESCAPE_ROUTES * escape_routes -
                        WALL_PENALTY * wall_penalty
                    )

                    # Determine if the new score is better than the previous best
                    if score > best_score:
                        best_score = score
                        best_move = d

            # Return what was found to be the best move - enforce np.array type
            return np.array(best_move)
        
        # If the manhattan distance from the pursuer is not within the specified range, go offensive
        path = a_star(world, tuple(current), predicted_pursued)
        if path and len(path) > 1:
            move = np.array(path[1]) - current_np
            return np.array(move) # Enforce np.array type
        
        # If no moves were found, stay still
        return np.array([0, 0])