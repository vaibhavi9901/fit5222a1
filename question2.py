"""
This is the python script for question 1. In this script, you are required to implement a single agent path-finding algorithm
"""

from lib_piglet.utils.tools import eprint
import glob, os, sys, heapq
from collections import deque


#import necessary modules that this python scripts need.
try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.utils.controller import get_action, Train_Actions, Directions, check_conflict, path_controller, evaluator, remote_evaluator
except Exception as e:
    eprint("Cannot load flatland modules!", e)
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

# ---------------------------------------------------------------------------
# Helper: BFS-based distance heuristic (ignores time and direction)
# Returns a dict mapping (x, y) -> min steps to goal, searching *backwards*
# from the goal so we can use it as an admissible heuristic for any position.
# ---------------------------------------------------------------------------
def bfs_heuristic(goal: tuple, rail: GridTransitionMap) -> dict:
    """
    Compute a backward BFS from `goal` on the undirected rail graph.
    Returns dist[(x, y)] = minimum number of steps from (x, y) to goal,
    ignoring direction (used as admissible lower bound).
    """
    dist = {goal: 0}
    queue = deque([goal])
 
    while queue:
        x, y = queue.popleft()
        # Try all four directions from this cell to find neighbours
        for d in range(4):
            transitions = rail.get_transitions(x, y, d)
            for action, valid in enumerate(transitions):
                if not valid:
                    continue
                nx, ny = x, y
                if action == Directions.NORTH:
                    nx -= 1
                elif action == Directions.EAST:
                    ny += 1
                elif action == Directions.SOUTH:
                    nx += 1
                elif action == Directions.WEST:
                    ny -= 1
                if (nx, ny) not in dist:
                    dist[(nx, ny)] = dist[(x, y)] + 1
                    queue.append((nx, ny))
    return dist
 
 
# ---------------------------------------------------------------------------
# Conflict checker against existing paths
# ---------------------------------------------------------------------------
def has_conflict(new_loc, cur_loc, t, existing_paths):
    """
    Returns True if moving from cur_loc -> new_loc at time t->t+1
    creates a vertex or edge conflict with any existing path.
 
    Vertex conflict : new_loc == existing_path[t+1]
    Edge   conflict : new_loc == existing_path[t] AND cur_loc == existing_path[t+1]
                      (two trains swapping positions)
    """
    for p in existing_paths:
        # Vertex conflict
        if t + 1 < len(p) and p[t + 1] == new_loc:
            return True
        # Edge conflict (swap)
        if (t + 1 < len(p) and t < len(p) and
                p[t + 1] == cur_loc and p[t] == new_loc):
            return True
        # After an existing path ends the train stays at its last location
        if t + 1 >= len(p) and len(p) > 0 and p[-1] == new_loc:
            return True
    return False
 
 
# ---------------------------------------------------------------------------
# Space-Time A*
# ---------------------------------------------------------------------------
def get_path(
    start: tuple,
    start_direction: int,
    goal: tuple,
    rail: GridTransitionMap,
    agent_id: int,
    existing_paths: list,
    max_timestep: int,
) -> list:
    """
    Space-Time A* with wait actions.
    State  : (x, y, direction, time)
    Action : move in any valid rail direction OR wait in place
    Returns a list of (x, y) tuples representing the path, or [] if infeasible.
    """
    # ---- Precompute admissible heuristic via backward BFS ------------------
    h_dist = bfs_heuristic(goal, rail)
 
    if start not in h_dist:
        # start is not connected to goal at all → no solution
        return []
 
    # ---- A* search ---------------------------------------------------------
    # Priority queue entry: (f, g, x, y, direction, time)
    # We store the full state including time so we can detect conflicts.
    start_h = h_dist.get(start, 0)
    # heap: (f, g, x, y, direction, time)
    open_heap = [(start_h, 0, start[0], start[1], start_direction, 0)]
    # visited: (x, y, direction, time) -> g-cost
    visited = {}
    # parent map for path reconstruction: (x,y,d,t) -> (x,y,d,t) or None
    parent = {(start[0], start[1], start_direction, 0): None}
 
    # Cap search at max_timestep to prevent infinite loops
    time_limit = int(max_timestep)
 
    while open_heap:
        f, g, x, y, direction, t = heapq.heappop(open_heap)
 
        state = (x, y, direction, t)
 
        # Skip if we already found a cheaper path to this state
        if state in visited and visited[state] <= g:
            continue
        visited[state] = g
 
        # ---- Goal check ----------------------------------------------------
        if (x, y) == goal:
            # Reconstruct path
            path = []
            cur = state
            while cur is not None:
                cx, cy, cd, ct = cur
                path.append((cx, cy))
                cur = parent[cur]
            path.reverse()
            return path
 
        if t >= time_limit:
            continue
 
        cur_loc = (x, y)
 
        # ---- Expand neighbours ---------------------------------------------
        # 1. Move actions: iterate over valid rail transitions
        transitions = rail.get_transitions(x, y, direction)
        for action, valid in enumerate(transitions):
            if not valid:
                continue
            nx, ny = x, y
            if action == Directions.NORTH:
                nx -= 1
            elif action == Directions.EAST:
                ny += 1
            elif action == Directions.SOUTH:
                nx += 1
            elif action == Directions.WEST:
                ny -= 1
 
            new_loc = (nx, ny)
 
            # Conflict check
            if has_conflict(new_loc, cur_loc, t, existing_paths):
                continue
 
            new_g = g + 1
            new_h = h_dist.get(new_loc, float('inf'))
            if new_h == float('inf'):
                continue  # unreachable cell
            new_f = new_g + new_h
            new_state = (nx, ny, action, t + 1)
 
            if new_state not in visited or visited[new_state] > new_g:
                parent[new_state] = state
                heapq.heappush(open_heap, (new_f, new_g, nx, ny, action, t + 1))
 
        # 2. Wait action: stay in place (direction unchanged)
        #    Only wait if we haven't reached the goal yet (saves SIC).
        wait_loc = cur_loc
        if not has_conflict(wait_loc, cur_loc, t, existing_paths):
            new_g = g + 1
            new_h = h_dist.get(wait_loc, float('inf'))
            new_f = new_g + new_h
            new_state = (x, y, direction, t + 1)
            if new_state not in visited or visited[new_state] > new_g:
                parent[new_state] = state
                heapq.heappush(open_heap, (new_f, new_g, x, y, direction, t + 1))
 
    # No solution found within time limit
    return []

#########################
# You should not modify codes below, unless you want to modify test_cases to test specific instance. You can read it know how we ran flatland environment.
########################
if __name__ == "__main__":
    if len(sys.argv) > 1:
        remote_evaluator(get_path,sys.argv)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path,"multi_test_case/level*_test_*.pkl"))
        if test_single_instance:
            test_cases = glob.glob(os.path.join(script_path,"multi_test_case/level{}_test_{}.pkl".format(level, test)))
        test_cases.sort()
        evaluator(get_path,test_cases,debug,visualizer,2)

