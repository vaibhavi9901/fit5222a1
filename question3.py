
from lib_piglet.utils.tools import eprint
from typing import List, Tuple
import glob, os, sys,time,json, heapq
from collections import deque

#import necessary modules that this python scripts need.
try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.envs.agent_utils import EnvAgent
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

def bfs_heuristic(goal: tuple, rail: GridTransitionMap) -> dict:
    """
    Backward BFS from goal (direction-agnostic).
    Returns {(x,y): min_steps_to_goal}. Admissible h for Space-Time A*.
    """
    dist = {goal: 0}
    q = deque([goal])
    while q:
        x, y = q.popleft()
        for d in range(4):
            for action, valid in enumerate(rail.get_transitions(x, y, d)):
                if not valid:
                    continue
                nx, ny = x, y
                if   action == Directions.NORTH: nx -= 1
                elif action == Directions.EAST:  ny += 1
                elif action == Directions.SOUTH: nx += 1
                elif action == Directions.WEST:  ny -= 1
                if (nx, ny) not in dist:
                    dist[(nx, ny)] = dist[(x, y)] + 1
                    q.append((nx, ny))
    return dist
 
 
def has_conflict(new_loc: tuple, cur_loc: tuple, t: int,
                 constraint_paths: list) -> bool:
    """
    Vertex and edge conflict check against already-planned paths.
    Agents removed at target are gone from the map, so we only block
    cells while the other agent's path is still active (t+1 < len(p)).
    """
    for p in constraint_paths:
        if not p:
            continue
        if t + 1 < len(p):
            if p[t + 1] == new_loc:                         # vertex conflict
                return True
            if p[t + 1] == cur_loc and p[t] == new_loc:    # edge (swap)
                return True
    return False
 
 
def space_time_astar(
    start: tuple,
    start_dir: int,
    goal: tuple,
    rail: GridTransitionMap,
    constraint_paths: list,
    max_timestep: int,
    h_dist: dict,
    start_time: int = 0,
) -> list:
    """
    Space-Time A* with wait actions.
 
    start_time: absolute timestep at which this agent begins its search.
    Conflict indices into constraint_paths are absolute timesteps.
 
    Returns list of (x,y) starting from start_time, or [] on failure.
    """
    if start not in h_dist:
        return []
 
    open_heap = [(h_dist[start], 0, start[0], start[1], start_dir, start_time)]
    visited: dict = {}
    parent: dict = {(start[0], start[1], start_dir, start_time): None}
 
    while open_heap:
        f, g, x, y, direction, t = heapq.heappop(open_heap)
        state = (x, y, direction, t)
 
        if state in visited and visited[state] <= g:
            continue
        visited[state] = g
 
        if (x, y) == goal:
            path = []
            cur = state
            while cur is not None:
                cx, cy, cd, ct = cur
                path.append((cx, cy))
                cur = parent[cur]
            path.reverse()
            return path
 
        if t >= max_timestep:
            continue
 
        cur_loc = (x, y)
 
        # Move actions
        for action, valid in enumerate(rail.get_transitions(x, y, direction)):
            if not valid:
                continue
            nx, ny = x, y
            if   action == Directions.NORTH: nx -= 1
            elif action == Directions.EAST:  ny += 1
            elif action == Directions.SOUTH: nx += 1
            elif action == Directions.WEST:  ny -= 1
 
            new_loc = (nx, ny)
            if has_conflict(new_loc, cur_loc, t, constraint_paths):
                continue
            new_h = h_dist.get(new_loc)
            if new_h is None:
                continue
            new_g = g + 1
            ns = (nx, ny, action, t + 1)
            if ns not in visited or visited[ns] > new_g:
                parent[ns] = state
                heapq.heappush(open_heap,
                               (new_g + new_h, new_g, nx, ny, action, t + 1))
 
        # Wait action
        if not has_conflict(cur_loc, cur_loc, t, constraint_paths):
            new_g = g + 1
            new_h = h_dist[cur_loc]
            ns = (x, y, direction, t + 1)
            if ns not in visited or visited[ns] > new_g:
                parent[ns] = state
                heapq.heappush(open_heap,
                               (new_g + new_h, new_g, x, y, direction, t + 1))
 
    return []
 
 
# ════════════════════════════════════════════════════════════════════════════
#  get_path — initial planning
# ════════════════════════════════════════════════════════════════════════════
 
def get_path(agents: List[EnvAgent], rail: GridTransitionMap,
             max_timestep: int) -> List[List[tuple]]:
    """
    Plan collision-free paths for all agents.
    Order: earliest deadline first so tight-deadline trains get priority
    access to the shortest paths, minimising penalty exposure.
    """
    n = len(agents)
    path_all = [[] for _ in range(n)]
 
    order = sorted(
        range(n),
        key=lambda i: (agents[i].deadline if agents[i].deadline is not None
                       else max_timestep)
    )
 
    planned: List[list] = []
 
    for agent_id in order:
        agent = agents[agent_id]
        h_dist = bfs_heuristic(agent.target, rail)
 
        path = space_time_astar(
            agent.initial_position,
            agent.initial_direction,
            agent.target,
            rail,
            planned,
            max_timestep,
            h_dist,
            start_time=0,
        )
 
        path_all[agent_id] = path
        planned.append(path)
 
    return path_all
 
 
# ════════════════════════════════════════════════════════════════════════════
#  replan — malfunction / collision recovery
# ════════════════════════════════════════════════════════════════════════════
 
def replan(
    agents: List[EnvAgent],
    rail: GridTransitionMap,
    current_timestep: int,
    existing_paths: List[List[tuple]],
    max_timestep: int,
    new_malfunction_agents: List[int],
    failed_agents: List[int],
) -> List[List[tuple]]:
    """
    Repair paths after a malfunction or collision.
 
    Critical invariant
    ------------------
    path_controller calls get_action(agent_id, path[t], env) whenever
    t < len(path) AND status != 3.  get_action reads agent.position, which
    is None for any agent that has never been spawned (status == 0).
    There is no way to spawn such an agent after t=0, so we must NOT extend
    their path — doing so would cause a TypeError on agent.position[0].
 
    Therefore: skip any agent whose position is None (status 0, not spawned)
    or who has already finished (status 2 / 3).
 
    Path layout for a replanned active agent
    -----------------------------------------
    new_path = existing_path[:current_timestep]    <- immutable history
             + [cur_pos] * (mal_dur + 1)           <- sit still during fault
             + suffix[1:]                          <- A* route to goal
                                                      (suffix[0] == cur_pos)
    Indices:  0 .. t-1  |  t .. t+mal_dur  |  t+mal_dur+1 ..
    """
    new_paths = [list(p) for p in existing_paths]
 
    replan_set = set(failed_agents) | set(new_malfunction_agents)
 
    # Earliest deadline first so tight-deadline agents get priority
    replan_order = sorted(
        replan_set,
        key=lambda i: (agents[i].deadline if agents[i].deadline is not None
                       else max_timestep)
    )
 
    for agent_id in replan_order:
        agent = agents[agent_id]
 
        # ── Skip agents we cannot or need not replan ─────────────────────
        # status 0 : not yet spawned — position is None, cannot be moved
        # status 2 : at goal (should be 3 with remove_agents_at_target=True)
        # status 3 : done and removed
        # position None : same as status 0 guard, defensive check
        if agent.status in (2, 3):
            continue
        if agent.position is None:
            # Cannot replan an unspawned agent — leave path as-is so the
            # path_controller keeps sending NOTHING (len(path) <= t).
            continue
 
        cur_pos = agent.position
        cur_dir = agent.direction
        mal_dur = (agent.malfunction_data.get("malfunction", 0)
                   if agent.malfunction_data else 0)
 
        # Already at goal — just extend with waits so path doesn't expire
        if cur_pos == agent.target:
            prefix = list(existing_paths[agent_id][:current_timestep])
            new_paths[agent_id] = prefix + [cur_pos] * (mal_dur + 1)
            continue
 
        # ── Build immutable history prefix ───────────────────────────────
        # Keep exactly current_timestep entries so path[current_timestep]
        # maps to the actual current position.
        raw_prefix = list(existing_paths[agent_id][:current_timestep])
        # If existing path was shorter than current_timestep (e.g. agent
        # had an empty path), pad with cur_pos so index alignment is right.
        if len(raw_prefix) < current_timestep:
            raw_prefix += [cur_pos] * (current_timestep - len(raw_prefix))
        prefix = raw_prefix
 
        # ── Forced wait segment (malfunction period) ─────────────────────
        # Covers absolute timesteps current_timestep .. current_timestep+mal_dur
        wait_segment = [cur_pos] * (mal_dur + 1)
 
        # ── Resume A* after malfunction clears ───────────────────────────
        resume_t = current_timestep + mal_dur
 
        # Build constraints from all OTHER agents' already-updated paths
        constraints = [new_paths[i] for i in range(len(agents))
                       if i != agent_id]
 
        h_dist = bfs_heuristic(agent.target, rail)
 
        suffix = space_time_astar(
            cur_pos, cur_dir, agent.target,
            rail, constraints, max_timestep, h_dist,
            start_time=resume_t,
        )
 
        if suffix:
            # suffix[0] == cur_pos at resume_t, already covered by
            # wait_segment[-1], so drop it to avoid duplication.
            new_paths[agent_id] = prefix + wait_segment + suffix[1:]
        else:
            # No path found — keep agent waiting; evaluator will penalise
            new_paths[agent_id] = prefix + wait_segment
 
    return new_paths


#####################################################################
# Instantiate a Remote Client
# You should not modify codes below, unless you want to modify test_cases to test specific instance.
#####################################################################
if __name__ == "__main__":

    if len(sys.argv) > 1:
        remote_evaluator(get_path,sys.argv, replan = replan)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path, "multi_test_case/level*_test_*.pkl"))

        if test_single_instance:
            test_cases = glob.glob(os.path.join(script_path,"multi_test_case/level{}_test_{}.pkl".format(level, test)))
        test_cases.sort()
        deadline_files =  [test.replace(".pkl",".ddl") for test in test_cases]
        evaluator(get_path, test_cases, debug, visualizer, 3, deadline_files, replan = replan)




