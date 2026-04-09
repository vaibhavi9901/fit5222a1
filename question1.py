# """
# This is the python script for question 1. In this script, you are required to implement a single agent path-finding algorithm
# """
# from lib_piglet.utils.tools import eprint
# import glob, os, sys
# import heapq

# #import necessary modules that this python scripts need.
# try:
#     from flatland.core.transition_map import GridTransitionMap
#     from flatland.utils.controller import get_action, Train_Actions, Directions, check_conflict, path_controller, evaluator, remote_evaluator
# except Exception as e:
#     eprint("Cannot load flatland modules!")
#     eprint(e)
#     exit(1)

# #########################
# # Debugger and visualizer options
# #########################

# # Set these debug option to True if you want more information printed
# debug = False
# visualizer = False

# # If you want to test on specific instance, turn test_single_instance to True and specify the level and test number
# test_single_instance = False
# level = 0
# test = 0


# #########################
# # Reimplementing the content in get_path() function.
# #
# # Return a list of (x,y) location tuples which connect the start and goal locations.
# #########################


# # This function return a list of location tuple as the solution.
# # @param start A tuple of (x,y) coordinates
# # @param start_direction An Int indicate direction.
# # @param goal A tuple of (x,y) coordinates
# # @param rail The flatland railway GridTransitionMap
# # @param max_timestep The max timestep of this episode.
# # @return path A list of (x,y) tuple.

# def heuristic(pos, goal):
#     """
#     Manhattan distance heuristic.
#     Admissible because each step moves exactly one cell in one cardinal direction.
#     """
#     return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
 
# # ------------ A* Heuristic ------------------

# def get_path(start: tuple, start_direction: int, goal: tuple, rail: GridTransitionMap, max_timestep: int):
#     """
#     A* search for single-agent pathfinding in the Flatland railway environment.
 
#     State space: (position, direction) — direction is part of the state because
#     valid transitions from a cell depend on the agent's current heading.
 
#     Returns a list of (x, y) location tuples from start to goal (inclusive).
#     Returns an empty list if no path is found.
#     """
 
#     # ---------------------------------------------------------------------------
#     # Direction deltas: North=0, East=1, South=2, West=3
#     # ---------------------------------------------------------------------------
#     direction_deltas = {
#         Directions.NORTH: (-1,  0),
#         Directions.EAST:  ( 0,  1),
#         Directions.SOUTH: ( 1,  0),
#         Directions.WEST:  ( 0, -1),
#     }


#     # ---------------------------------------------------------------------------
#     # A* open list: (f, g, position, direction, parent_key)
#     # We use a heap keyed on f = g + h.
#     # ---------------------------------------------------------------------------
#     start_state = (start, start_direction)
#     h = heuristic(start, goal)
#     # heap entry: (f_cost, g_cost, position, direction)
#     open_heap = [(h, 0, start, start_direction)]
 
#     # g_cost: best known cost to reach each (position, direction) state
#     g_cost = {start_state: 0}
 
#     # came_from: maps each state to its parent state, used to reconstruct path
#     came_from = {start_state: None}
 
#     while open_heap:
#         f, g, pos, direction = heapq.heappop(open_heap)
 
#         # Skip if we've already found a cheaper way to this state
#         if g > g_cost.get((pos, direction), float('inf')):
#             continue
 
#         # Goal check — reached the goal cell
#         if pos == goal:
#             return reconstruct_path(came_from, (pos, direction))
 
#         # Expand neighbours using valid rail transitions
#         valid_transitions = rail.get_transitions(pos[0], pos[1], direction)
 
#         for new_direction, is_valid in enumerate(valid_transitions):
#             if not is_valid:
#                 continue
 
#             dx, dy = direction_deltas[new_direction]
#             new_pos = (pos[0] + dx, pos[1] + dy)
 
#             new_g = g + 1  # uniform step cost
#             new_state = (new_pos, new_direction)
 
#             if new_g < g_cost.get(new_state, float('inf')):
#                 g_cost[new_state] = new_g
#                 came_from[new_state] = (pos, direction)
#                 new_f = new_g + heuristic(new_pos, goal)
#                 heapq.heappush(open_heap, (new_f, new_g, new_pos, new_direction))
 
#     # No path found
#     return []
 
 
# def reconstruct_path(came_from, final_state):
#     """
#     Walk back through came_from to reconstruct the path.
#     Returns a list of (x, y) position tuples from start to goal.
#     """
#     path = []
#     state = final_state
#     while state is not None:
#         pos, _ = state
#         path.append(pos)
#         state = came_from[state]
#     path.reverse()
#     return path

# # # ------- Jump point search ---------------
# # def get_path(start: tuple, start_direction: int, goal: tuple, rail: GridTransitionMap, max_timestep: int):
# #     """
# #     Jump Point Search (JPS) for single-agent pathfinding in the Flatland railway environment.
 
# #     JPS Reference:
# #         Harabor & Grastien (2011), "Online Graph Pruning for Pathfinding
# #         on Grid Maps", AAAI-11.
 
# #     JPS is an optimisation of A* for uniform-cost grid graphs. It reduces
# #     open-list insertions by skipping intermediate nodes on straight rail
# #     corridors, only adding jump points to the open list.
 
# #     KEY DESIGN DECISION — full path tracking during jump():
# #         Classic JPS on open grids interpolates straight lines between jump
# #         points during reconstruction because movement is always geometrically
# #         straight. In Flatland, rails constrain movement so the path between
# #         two jump points may not be a geometric straight line. To ensure the
# #         reconstructed path is valid and optimal, jump() records every
# #         intermediate cell it visits. came_from therefore stores the complete
# #         segment of cells between consecutive jump points, not just the
# #         jump point endpoints. This guarantees the final reconstructed path
# #         follows actual rail transitions.
 
# #     IMPORTANT NOTE ON START CELL:
# #         initial_direction is the direction the agent faces, not the entry
# #         direction used by get_transitions(). At the start cell we therefore
# #         try all 4 entry directions to find every valid first move.
 
# #     NOTE ON JPS IN FLATLAND:
# #         Flatland's rail topology constrains most cells to 1-2 valid
# #         transitions, so most cells are already jump points by definition.
# #         JPS pruning gains are therefore more modest than on open grids, but
# #         long straight track segments are still skipped entirely.
 
# #     State space : (position, direction) -- direction affects valid transitions
# #     Heuristic   : Manhattan distance    -- admissible and consistent
# #     Cost        : uniform (1 per step)
 
# #     Returns a list of (x, y) location tuples from start to goal (inclusive).
# #     Returns an empty list if no path is found.
# #     """
 
# #     # ---------------------------------------------------------------------------
# #     # Direction deltas: North=0, East=1, South=2, West=3
# #     # ---------------------------------------------------------------------------
# #     direction_deltas = {
# #         Directions.NORTH: (-1,  0),
# #         Directions.EAST:  ( 0,  1),
# #         Directions.SOUTH: ( 1,  0),
# #         Directions.WEST:  ( 0, -1),
# #     }
 
# #     opposite = {
# #         Directions.NORTH: Directions.SOUTH,
# #         Directions.SOUTH: Directions.NORTH,
# #         Directions.EAST:  Directions.WEST,
# #         Directions.WEST:  Directions.EAST,
# #     }
 
# #     # ---------------------------------------------------------------------------
# #     # Helper: all valid (new_pos, new_dir) moves from pos given current heading
# #     # ---------------------------------------------------------------------------
# #     def get_neighbours(pos, direction):
# #         neighbours = []
# #         transitions = rail.get_transitions(pos[0], pos[1], direction)
# #         for new_dir, is_valid in enumerate(transitions):
# #             if not is_valid:
# #                 continue
# #             if new_dir == opposite[direction]:
# #                 continue  # no U-turns
# #             dx, dy = direction_deltas[new_dir]
# #             neighbours.append(((pos[0] + dx, pos[1] + dy), new_dir))
# #         return neighbours
 
# #     # ---------------------------------------------------------------------------
# #     # Start cell: try all 4 entry directions to find every valid first move.
# #     # This handles the mismatch between initial_direction and entry direction.
# #     # ---------------------------------------------------------------------------
# #     def get_start_neighbours(pos):
# #         seen = set()
# #         neighbours = []
# #         for entry_dir in range(4):
# #             transitions = rail.get_transitions(pos[0], pos[1], entry_dir)
# #             for new_dir, is_valid in enumerate(transitions):
# #                 if not is_valid:
# #                     continue
# #                 dx, dy = direction_deltas[new_dir]
# #                 new_pos = (pos[0] + dx, pos[1] + dy)
# #                 key = (new_pos, new_dir)
# #                 if key not in seen:
# #                     seen.add(key)
# #                     neighbours.append((new_pos, new_dir))
# #         return neighbours
 
# #     # ---------------------------------------------------------------------------
# #     # Jump: probe straight in `direction` from `pos`, recording every cell.
# #     #
# #     # Returns (jump_point_pos, intermediate_cells) where intermediate_cells
# #     # is the list of every cell visited along the way (excluding pos itself,
# #     # including the jump point). Returns None if the path is blocked.
# #     #
# #     # We record intermediate cells rather than just step count because
# #     # Flatland rails are not always geometrically straight -- the actual
# #     # path must follow rail transitions, not interpolated straight lines.
# #     # ---------------------------------------------------------------------------
# #     def jump(pos, direction):
# #         current = pos
# #         intermediate = []
 
# #         while True:
# #             # Check straight move is valid from current cell
# #             transitions = rail.get_transitions(current[0], current[1], direction)
# #             if not transitions[direction]:
# #                 return None  # rail blocks straight movement
 
# #             # Step forward and record the cell
# #             dx, dy = direction_deltas[direction]
# #             current = (current[0] + dx, current[1] + dy)
# #             intermediate.append(current)
 
# #             # Goal is always a jump point
# #             if current == goal:
# #                 return (current, intermediate)
 
# #             # Count onward transitions (excluding U-turn)
# #             next_transitions = rail.get_transitions(current[0], current[1], direction)
# #             onward = sum(
# #                 1 for d, v in enumerate(next_transitions)
# #                 if v and d != opposite[direction]
# #             )
 
# #             if onward > 1:
# #                 # Junction -- forced neighbours exist, must stop here
# #                 return (current, intermediate)
 
# #             if onward == 0:
# #                 # Dead end in this direction
# #                 return None
 
# #             # onward == 1: straight corridor, keep probing
 
# #     # ---------------------------------------------------------------------------
# #     # Identify successors using JPS rules.
# #     # Returns list of (successor_pos, successor_dir, new_g, segment).
# #     # segment is the list of cells from current pos to successor (inclusive)
# #     # -- stored so reconstruct_path can expand jump point chains into full paths.
# #     # ---------------------------------------------------------------------------
# #     def identify_successors(pos, direction, g, is_start=False):
# #         successors = []
# #         neighbours = get_start_neighbours(pos) if is_start else get_neighbours(pos, direction)
 
# #         for next_pos, next_dir in neighbours:
# #             # Goal reachable in one step
# #             if next_pos == goal:
# #                 successors.append((next_pos, next_dir, g + 1, [next_pos]))
# #                 continue
 
# #             if next_dir == direction or is_start:
# #                 # Straight move -- attempt to jump over the corridor
# #                 result = jump(next_pos, next_dir)
# #                 if result is not None:
# #                     jp_pos, intermediate = result
# #                     # intermediate includes next_pos through jp_pos
# #                     full_segment = [next_pos] + intermediate[:-1] if next_pos != intermediate[0] else intermediate
# #                     # Simpler: build segment as next_pos + everything jump recorded
# #                     segment = [next_pos] + [c for c in intermediate if c != next_pos]
# #                     new_g = g + len(segment)
# #                     successors.append((jp_pos, next_dir, new_g, segment))
# #                 else:
# #                     # Cannot jump further -- add next_pos directly
# #                     successors.append((next_pos, next_dir, g + 1, [next_pos]))
# #             else:
# #                 # Turn -- next_pos is itself a jump point
# #                 successors.append((next_pos, next_dir, g + 1, [next_pos]))
 
# #         return successors
 
# #     # ---------------------------------------------------------------------------
# #     # Path reconstruction using stored segments.
# #     # came_from maps each state -> (parent_state, segment_to_here)
# #     # We walk back through parents collecting segments, then reverse and flatten.
# #     # ---------------------------------------------------------------------------
# #     def reconstruct_path(came_from, final_state):
# #         segments = []
# #         state = final_state
# #         while came_from[state] is not None:
# #             parent_state, segment = came_from[state]
# #             segments.append(segment)
# #             state = parent_state
# #         segments.reverse()
# #         # Start position + all segments flattened
# #         full_path = [state[0]]  # state is now the start state
# #         for seg in segments:
# #             full_path.extend(seg)
# #         return full_path
 
# #     # ---------------------------------------------------------------------------
# #     # JPS main loop (A* framework with JPS successor generation)
# #     # ---------------------------------------------------------------------------
# #     if start == goal:
# #         return [start]
 
# #     start_state = (start, start_direction)
# #     open_heap = [(heuristic(start, goal), 0, start, start_direction)]
# #     g_cost = {start_state: 0}
# #     # came_from maps state -> (parent_state, segment) or None for start
# #     came_from = {start_state: None}
 
# #     while open_heap:
# #         f, g, pos, direction = heapq.heappop(open_heap)
 
# #         # Skip stale heap entries
# #         if g > g_cost.get((pos, direction), float('inf')):
# #             continue
 
# #         # Goal reached
# #         if pos == goal:
# #             return reconstruct_path(came_from, (pos, direction))
 
# #         # Expand using JPS successors
# #         is_start = (pos == start)
# #         for jp_pos, jp_dir, new_g, segment in identify_successors(pos, direction, g, is_start):
# #             new_state = (jp_pos, jp_dir)
# #             if new_g < g_cost.get(new_state, float('inf')):
# #                 g_cost[new_state] = new_g
# #                 came_from[new_state] = ((pos, direction), segment)
# #                 new_f = new_g + heuristic(jp_pos, goal)
# #                 heapq.heappush(open_heap, (new_f, new_g, jp_pos, jp_dir))
 
# #     return []


# #########################
# # You should not modify codes below, unless you want to modify test_cases to test specific instance. You can read it know how we ran flatland environment.
# ########################
# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         remote_evaluator(get_path,sys.argv)
#     else:
#         script_path = os.path.dirname(os.path.abspath(__file__))
#         test_cases = glob.glob(os.path.join(script_path,"single_test_case/level*_test_*.pkl"))
#         if test_single_instance:
#             test_cases = glob.glob(os.path.join(script_path,"single_test_case/level{}_test_{}.pkl".format(level, test)))
#         test_cases.sort()
#         evaluator(get_path,test_cases,debug,visualizer,1)

