import numpy as np, random, math
from typing import Tuple, Optional

# Define Available Directions
DIRECTIONS = [(0, 0),                                # Staying still
              (-1, 0), (1, 0), (0, -1), (0, 1),      # Up, down, left, right
              (-1, -1), (-1, 1), (1, -1), (1, 1)]    # Diagonal Moves

# Define a distance at which to begin evading/attacking if in unoptimal position
REACT_DISTANCE = 3

# Required number of samples before modifying our intended actions
REQUIRED_SAMPLES = 30

# Define a distance at which rollout reward = 0 & at which X penalty is applied to the reward
SEVERE_ROLLOUT_DANGER_DISTANCE = 2
ROLLOUT_DANGER_DISTANCE = 4
ROLLOUT_DANGER_PENALTY = 0.5

# Define an escape route penalty - penalizes the route if not many escape routes
ESCAPE_ROUTE_PENALTY = 0.3

# Define the limits for MCTS
MCTS_ITER_LIMIT = 150
MCTS_MAX_DEPTH = 5

# Global Tracking Variables
modification_counts = {"LEFT": 1, "STRAIGHT": 1, "RIGHT": 1}
intended_location = np.array([0, 0])
intended_direction = np.array([0, 0])


###################
# MCTS Node Class #
###################
class MCTSNode:
    def __init__(self, direction, parent=None):
        self.direction = direction
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0

    # Define a function to check if the node is fully expanded
    def is_fully_expanded(self):
        return len(self.children) == len(DIRECTIONS)
    
    # Define a function to find the best child in this node
    #   c_param defines the preference for exploit vs. explore
    #       1.4 default results in exploring but not too much
    def best_child(self, c_param=1.4):
        # Define initial variables
        best_score = -float("inf")
        best = None

        # Loop through all of the children
        for _, child in self.children.items():
            # Calculate the score using the UCB1 formula: (w_i / n_i) + c * sqrt((ln(N_i)) / n_i)
            score = (float("inf") if child.visits == 0 else
                     (child.value / child.visits) + c_param * math.sqrt(math.log(self.visits + 1) / child.visits))
            
            # If new best score was found, update tracker variables
            if score > best_score:
                best_score = score
                best = child

        return best
    

####################
# Helper Functions #
####################
# Define a function to return Manhattan distance
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Define a function to check if the current move is a valid one
def is_valid(world, state):
    row, col = state
    return 0 <= row < len(world) and 0 <= col < len(world[0]) and world[row][col] == 0

# Define a function that returns the new location, given a direction to apply
#   Returns none if is an invalid new location
def apply_direction(world, current, direction):
    new_location = (current[0] + direction[0], current[1] + direction[1])
    return new_location if is_valid(world, new_location) else None

# Define a function to get the probability of each modification (LEFT, STRAIGHT, RIGHT)
def get_modification_probabilities():
    total = sum(modification_counts.values())
    return {modification: count / total for modification, count in modification_counts.items()}

# Define a function to check if the intended move is what actually occurred
def check_intended(current, intended_location, intended_direciton, previous_location):
    # Calculate the difference between current and intended
    difference = current - intended_location

    # If difference is not (0, 0) then determine what happened (LEFT/RIGHT)
    # Else, our desired action took place, increment STRAIGHT
    if not np.array_equal(difference, (0, 0)):
        # Determine the actual direction taken
        actual_direction = current - previous_location

        # If it was LEFT, increment LEFT
        # Else, if it was RIGHT, increment RIGHT
        if np.array_equal(actual_direction, (-intended_direciton[1], intended_direciton[0])):
            modification_counts["LEFT"] += 1
        elif np.array_equal(actual_direction, (intended_direciton[1], -intended_direciton[0])):
            modification_counts["RIGHT"] += 1
    else:
        modification_counts["STRAIGHT"] += 1

# Define a function to modify the intended direction
def modify_direction(world, current, direction, probability):
    # Get left/right movements
    left = np.array([-direction[1], direction[0]])
    right = np.array([direction[1], -direction[0]])
    # If probability of LEFT is greater than STRAIGHT, and is valid, move left instead
    # Else, if probability of RIGHT is greater than STRAIGHT, and is valid, move right instead
    if probability["LEFT"] > probability["STRAIGHT"] and apply_direction(world, current, left) is not None:
        return left
    elif probability["RIGHT"] > probability["STRAIGHT"] and apply_direction(world, current, right) is not None:
        return right
    return direction

# Define a function that simulates the rollout of the MCTS
def simulate_rollout(world, current, pursued, pursuer, mod_probs, direction, prev_state):
    # If not valid, return 0
    if not is_valid(world, current):
        return 0.0
    
    # Calculate the starting reward - uses Chebyshev distance
    reward = 3.0 - min(1.0, max(abs(current[0] - pursued[0]), abs(current[1] - pursued[1])) / 10.0)

    # Provide a penalty for being near the pursuer - uses Chebyshev distance
    danger_dist = max(abs(current[0] - pursuer[0]), abs(current[1] - pursuer[1]))
    # If right next to pursuer, return 0
    # Else, if equal to two, penalty of X
    if danger_dist <= SEVERE_ROLLOUT_DANGER_DISTANCE:
        return 0.0
    elif danger_dist <= ROLLOUT_DANGER_DISTANCE:
        reward -= ROLLOUT_DANGER_PENALTY

    # Penalize based on how many surrounding directions are invalid
    escape_routes = 0
    # Loop through all directions, except standing still
    for dir in DIRECTIONS[1:]:
        neighbor = (current[0] + dir[0], current[1] + dir[1])
        # Check if the new is valid
        if is_valid(world, neighbor):
            escape_routes += 1

    # Penalize the route for less escape routes
    max_possible_routes = len(DIRECTIONS) - 1
    escape_penalty = (max_possible_routes - escape_routes) / max_possible_routes
    reward -= 0.3 * escape_penalty

    # Define dict storing the three possible moves
    dirs = {
        "LEFT": (-direction[1], direction[0]),
        "STRAIGHT": direction,
        "RIGHT": (direction[1], -direction[1])
    }

    # Loop through the different directions
    for mod_direction, mod_move in dirs.items():
        # Calculate the new location
        new_location = (prev_state[0] + mod_move[0], prev_state[1] + mod_move[1])
        # If not valid, give a penalty equal to the probability
        if not is_valid(world, new_location):
            reward -= mod_probs.get(mod_direction, 0.3)
        
    # Return at least 0
    return max(0.0, reward)

# Define a function to traverse the nodes
def mcts_traverse(world, node, state, depth):
    # Stop if max depth reached or not fully expanded
    if depth >= MCTS_MAX_DEPTH or not node.is_fully_expanded():
        return node, state
    
    # Get best child
    child = node.best_child()
    if child is None:
        return node, state
    
    # Get the new location and check if it's valid
    next_state = apply_direction(world, state, child.direction)
    if next_state is None:
        return node, state
    
    # Recursively traverse the nodes
    return mcts_traverse(world, child, next_state, depth + 1)

# Define a function to perform the MCTS search
def mcts_search(world, initial_state, pursued, pursuer, mod_probs, iter_limit):
    # Define a root node
    root = MCTSNode(direction=(0, 0))

    # Loop a max of iter_limit number of times
    for _ in range(iter_limit):
        ### Tree Traversal ###
        node, sim_state = mcts_traverse(world, root, initial_state, 0)

        ### Expansion ###
        # Get the untried directions
        untried = [dir for dir in DIRECTIONS if dir not in node.children and apply_direction(world, sim_state, dir)]
        # If untried, try them
        # Else, set reward based on sim_state
        if untried:
            direction = random.choice(untried)
            new_state = apply_direction(world, sim_state, direction)
            # If new state is invalid, continue
            if new_state is None:
                continue
            # Get child node and simulate rollout
            child = MCTSNode(direction=direction, parent=node)
            node.children[direction] = child
            reward = simulate_rollout(world, new_state, pursued, pursuer, mod_probs, direction, sim_state)
        else:
            reward = simulate_rollout(world, sim_state, pursued, pursuer, mod_probs, node.direction, sim_state)
            child = node

        ### Backpropagation ###
        # While there is a child, update values
        while child:
            child.visits += 1
            child.value += reward
            child = child.parent

    # Get a list of all safe (valid) children
    safe_children = {
        direction: node for direction, node in root.children.items()
        if apply_direction(world, initial_state, direction) is not None
    }

    # If no safe children, return (0, 0)
    if not safe_children:
        return np.array([0, 0])
    
    # Determine the best possible action - (1e-5) ensures division by 0 does not occur
    best_action = max(safe_children.items(), key=lambda item: item[1].value / (item[1].visits + 1e-5))[0]
    return np.array(best_action)


######################
# PlannerAgent Class #
######################
class PlannerAgent:
    def __init__(self):
        self.previous_location = np.array([0, 0])

    # Define function to check if all three directions are valid (LEFT, STRAIGHT, RIGHT)
    def all_directions_valid(self, world, current, base_dir):
        # Get left/right dirs
        left, right = (-base_dir[1], base_dir[0]), (base_dir[1], -base_dir[0])
        return all(is_valid(world, tuple(np.array(current) + np.array(curr_dir))) for curr_dir in [base_dir, left, right])

    # Define the main, plan_action function that returns the next best move
    def plan_action(self, world: np.ndarray, current: Tuple[int, int],
                    pursued: Tuple[int, int], pursuer: Tuple[int, int]) -> Optional[np.ndarray]:
        # Pull global variables
        global intended_direction, intended_location

        # Check intended, if it wasn't to stay still
        if not np.array_equal(intended_direction, (0, 0)):
            check_intended(current, intended_location, intended_direction, self.previous_location)

        # Get new probabilities, intended direction, and update previous location
        mod_probs = get_modification_probabilities()
        intended_direction = mcts_search(world, tuple(current), tuple(pursued), tuple(pursuer), mod_probs, MCTS_ITER_LIMIT)
        self.previous_location = np.array(current)

        # If at least X sample values have been obtained, modify action based on probabilities
        if sum(modification_counts.values()) > REQUIRED_SAMPLES:
            intended_direction = modify_direction(world, current, intended_direction, mod_probs)

        # Option 1: Try original intended direction IF all three possible directions are valid
        if self.all_directions_valid(world, current, tuple(intended_direction)):
            intended_location = np.array(current) + intended_direction
            return intended_direction
        
        # Option 2: Try diagonals IF any of previous three were not valid
        diagonals = DIRECTIONS[5:]
        current_np, pursued_np = np.array(current), np.array(pursued)
        diagonals.sort(key=lambda d: np.sum(np.abs((current_np + d) - pursued_np)))

        # Loop through diagonals
        for d in diagonals:
            # If not valid, continue
            if not is_valid(world, tuple(current + d)):
                continue
            # If all three directions are valid, return it
            if self.all_directions_valid(world, current, d):
                intended_location = np.array(current) + np.array(d)
                return np.array(d)
            
        # Option 3: Stay still
        intended_direction = np.array([0, 0])
        intended_location = np.array(current)

        # Option 4: If in danger or goal is nearby, avoid or chase
        if manhattan(current, pursuer) <= REACT_DISTANCE:
            # Evade by increasing distance from pursuer
            best = (0, 0)
            best_dist = manhattan(current, pursuer)
            # Loop through all directions, except staying still
            for dir in DIRECTIONS[1:]:
                # Calculate new location
                new_location = tuple(current_np + np.array(dir))
                # If is valid move, check if new best move
                if is_valid(world, new_location):
                    dist = manhattan(new_location, pursuer)
                    if dist > best_dist:
                        best = dir
                        best_dist = dist

            intended_direction = np.array(best)
            intended_location = current_np + intended_direction
            return np.array(best)
        
        if manhattan(current, pursued) <= REACT_DISTANCE:
            # Chase by closing the distance
            best = (0, 0)
            best_dist = manhattan(current, pursued)
            # Loop through all directions, except staying still
            for dir in DIRECTIONS[1:]:
                # Calculate new location
                new_location = tuple(current_np + np.array(dir))
                # If is valid move, check if new best move
                if is_valid(world, new_location):
                    dist = manhattan(new_location, pursued)
                    if dist < best_dist:
                        best = dir
                        best_dist = dist

            intended_direction = np.array(best)
            intended_location = current_np + intended_direction
            return np.array(best)
        
        return np.array([0, 0])