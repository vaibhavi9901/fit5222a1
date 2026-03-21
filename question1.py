"""
This is the python script for question 1. In this script, you are required to implement a single agent path-finding algorithm
"""
from lib_piglet.utils.tools import eprint
import glob, os, sys
import heapq

#import necessary modules that this python scripts need.
try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.utils.controller import get_action, Train_Actions, Directions, check_conflict, path_controller, evaluator, remote_evaluator
except Exception as e:
    eprint("Cannot load flatland modules!")
    eprint(e)
    exit(1)

#########################
# Debugger and visualizer options
#########################

# Set these debug option to True if you want more information printed
debug = False
visualizer = False

# If you want to test on specific instance, turn test_single_instance to True and specify the level and test number
test_single_instance = False
level = 0
test = 0


#########################
# Reimplementing the content in get_path() function.
#
# Return a list of (x,y) location tuples which connect the start and goal locations.
#########################


# This function return a list of location tuple as the solution.
# @param start A tuple of (x,y) coordinates
# @param start_direction An Int indicate direction.
# @param goal A tuple of (x,y) coordinates
# @param rail The flatland railway GridTransitionMap
# @param max_timestep The max timestep of this episode.
# @return path A list of (x,y) tuple.
def heuristic(pos, goal):
    """
    Manhattan distance heuristic.
    Admissible because each step moves exactly one cell in one cardinal direction.
    """
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
 
 
def get_path(start: tuple, start_direction: int, goal: tuple, rail: GridTransitionMap, max_timestep: int):
    """
    A* search for single-agent pathfinding in the Flatland railway environment.
 
    State space: (position, direction) — direction is part of the state because
    valid transitions from a cell depend on the agent's current heading.
 
    Returns a list of (x, y) location tuples from start to goal (inclusive).
    Returns an empty list if no path is found.
    """
 
    # ---------------------------------------------------------------------------
    # Direction deltas: North=0, East=1, South=2, West=3
    # ---------------------------------------------------------------------------
    direction_deltas = {
        Directions.NORTH: (-1,  0),
        Directions.EAST:  ( 0,  1),
        Directions.SOUTH: ( 1,  0),
        Directions.WEST:  ( 0, -1),
    }
 
    # ---------------------------------------------------------------------------
    # A* open list: (f, g, position, direction, parent_key)
    # We use a heap keyed on f = g + h.
    # ---------------------------------------------------------------------------
    start_state = (start, start_direction)
    h = heuristic(start, goal)
    # heap entry: (f_cost, g_cost, position, direction)
    open_heap = [(h, 0, start, start_direction)]
 
    # g_cost: best known cost to reach each (position, direction) state
    g_cost = {start_state: 0}
 
    # came_from: maps each state to its parent state, used to reconstruct path
    came_from = {start_state: None}
 
    while open_heap:
        f, g, pos, direction = heapq.heappop(open_heap)
 
        # Skip if we've already found a cheaper way to this state
        if g > g_cost.get((pos, direction), float('inf')):
            continue
 
        # Goal check — reached the goal cell
        if pos == goal:
            return reconstruct_path(came_from, (pos, direction))
 
        # Expand neighbours using valid rail transitions
        valid_transitions = rail.get_transitions(pos[0], pos[1], direction)
 
        for new_direction, is_valid in enumerate(valid_transitions):
            if not is_valid:
                continue
 
            dx, dy = direction_deltas[new_direction]
            new_pos = (pos[0] + dx, pos[1] + dy)
 
            new_g = g + 1  # uniform step cost
            new_state = (new_pos, new_direction)
 
            if new_g < g_cost.get(new_state, float('inf')):
                g_cost[new_state] = new_g
                came_from[new_state] = (pos, direction)
                new_f = new_g + heuristic(new_pos, goal)
                heapq.heappush(open_heap, (new_f, new_g, new_pos, new_direction))
 
    # No path found
    return []
 
 
def reconstruct_path(came_from, final_state):
    """
    Walk back through came_from to reconstruct the path.
    Returns a list of (x, y) position tuples from start to goal.
    """
    path = []
    state = final_state
    while state is not None:
        pos, _ = state
        path.append(pos)
        state = came_from[state]
    path.reverse()
    return path

#########################
# You should not modify codes below, unless you want to modify test_cases to test specific instance. You can read it know how we ran flatland environment.
########################
if __name__ == "__main__":
    if len(sys.argv) > 1:
        remote_evaluator(get_path,sys.argv)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path,"single_test_case/level*_test_*.pkl"))
        if test_single_instance:
            test_cases = glob.glob(os.path.join(script_path,"single_test_case/level{}_test_{}.pkl".format(level, test)))
        test_cases.sort()
        evaluator(get_path,test_cases,debug,visualizer,1)



















