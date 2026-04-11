
from lib_piglet.utils.tools import eprint
from typing import List, Tuple, Dict, Optional
import glob, os, sys,time,json, heapq, random
from collections import deque
import multiprocessing as mp

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
debug = True
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
    time_limit: float = None,  # NEW: time limit in seconds
) -> list:
    """
    Space-Time A* with wait actions.
 
    start_time: absolute timestep at which this agent begins its search.
    Conflict indices into constraint_paths are absolute timesteps.
 
    Returns list of (x,y) starting from start_time, or [] on failure.
    """
    if start not in h_dist:
        return []
 
    # Record start time for timeout checking
    search_start_time = time.time()

    open_heap = [(h_dist[start], 0, start[0], start[1], start_dir, start_time)]
    visited: dict = {}
    parent: dict = {(start[0], start[1], start_dir, start_time): None}
 
    while open_heap:
        # Check time limit BEFORE each node expansion
        if time_limit is not None and (time.time() - search_start_time) > time_limit:
            # Timeout - return empty path
            return []
        
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

############ LNS-based planning ###############

# ── LNS parameters ───────────────────────────────────────────────────────────
LNS_ITERATIONS_INITIAL  = 1   # iterations in get_path
LNS_NEIGHBOURHOOD_SIZE  = 5     # agents per LNS neighbourhood
LNS_ITERATIONS_REPLAN   = 50    # iterations per replan call
LNS_TIME_BUDGET         = 20.0  # seconds budget for LNS in get_path
LNS_TIME_BUDGET_REPLAN  = 1.0   # seconds per replan call

def precompute_heuristics(agents: List[EnvAgent],
                          rail: GridTransitionMap) -> List[dict]:
    """BFS heuristic for every agent, cached by goal cell."""
    cache: Dict[tuple, dict] = {}
    result = []
    for agent in agents:
        if agent.target not in cache:
            cache[agent.target] = bfs_heuristic(agent.target, rail)
        result.append(cache[agent.target])
    return result

# ════════════════════════════════════════════════════════════════════════════
#  Slack-based priority ordering   (Chen et al. §"Slack Based Priority")
# ════════════════════════════════════════════════════════════════════════════

def compute_slack_order(agents, h_dists, max_timestep):
    """
    slack_i = deadline_i - earliest_possible_arrival_i
    Uses already-computed h_dists — does NOT rerun BFS.
    """
    slacks = []
    for i, agent in enumerate(agents):
        earliest = h_dists[i].get(agent.initial_position, max_timestep)
        ddl = agent.deadline if agent.deadline is not None else max_timestep
        slack = ddl - earliest
        slacks.append((slack, earliest, i))
    slacks.sort()
    return [i for _, _, i in slacks]
 
 
# ════════════════════════════════════════════════════════════════════════════
#  Delay metric
# ════════════════════════════════════════════════════════════════════════════
 
def agent_delay(agent, path, max_timestep):
    if not path:
        return max_timestep
    ddl = agent.deadline if agent.deadline is not None else max_timestep
    return max(0, len(path) - 1 - ddl)
 
 
def total_delay(agents, paths, max_timestep):
    return sum(agent_delay(agents[i], paths[i], max_timestep)
               for i in range(len(agents)))
 
 
# ════════════════════════════════════════════════════════════════════════════
#  LNS neighbourhood  — FIX: windowed scan, not full path scan
# ════════════════════════════════════════════════════════════════════════════
 
def find_blocking_agents(focus_id, paths, neighbourhood_size, start_time=0):
    focus_path = paths[focus_id]
    if not focus_path:
        return [focus_id]
 
    # FIX: only scan a 60-step window, not the entire path
    window_start = start_time
    window_end   = min(len(focus_path), start_time + 60)
    focus_cells  = set(focus_path[window_start:window_end])
 
    overlap_count = {}
    for other_id, p in enumerate(paths):
        if other_id == focus_id or not p:
            continue
        for cell in p[window_start:min(len(p), window_end)]:
            if cell in focus_cells:
                overlap_count[other_id] = overlap_count.get(other_id, 0) + 1
 
    ranked = sorted(overlap_count.items(), key=lambda x: -x[1])
    return [focus_id] + [aid for aid, _ in ranked[:neighbourhood_size - 1]]
 
# ════════════════════════════════════════════════════════════════════════════
#  Adaptive parameters based on instance properties
# ════════════════════════════════════════════════════════════════════════════
 
def adapt_parameters(agents, h_dists, max_timestep):
    """
    Detect instance difficulty and return adapted parameters.
    Avoids the two extremes: neighbourhood too large (slow) or too small
    (can't escape local optima).
    """
    n = len(agents)
    slacks, dists = [], []
    for i, agent in enumerate(agents):
        dist = h_dists[i].get(agent.initial_position, max_timestep)
        ddl  = agent.deadline if agent.deadline is not None else max_timestep
        slacks.append(ddl - dist)
        dists.append(dist)
 
    min_slack  = min(slacks)
    tight_ratio = sum(1 for s in slacks if s < 20) / n
    mean_dist  = sum(dists) / n
 
    # Neighbourhood: 5 for large instances, up to 7 for small ones
    nbr = min(20, max(5, n//10))
    
    # if n <= 12:
    #     nbr = 7
    # elif n <= 30:
    #     nbr = 6
    # else:
    #     nbr = 5          # never set to n — that's the bug we're fixing
 
    # Iterations: fewer for large/hard instances (each call is slower)
    if n <= 12:
        iters_initial = 20
        iters_replan  = 50
        lns_budget    = max_timestep #25.0
        replan_budget = 2.0
    elif n <= 30:
        iters_initial = 10
        iters_replan  = 50
        lns_budget    = max_timestep #15.0
        replan_budget = 1.5
    else:
        iters_initial = 20
        iters_replan  = 50
        lns_budget    = max_timestep #12.0
        replan_budget = 1.0
 
    # A* time limit per call scales with mean path length
    astar_limit = min(2.0, max(0.5, mean_dist / 30.0))
 
    return dict(
        neighbourhood_size  = nbr,
        iters_initial       = iters_initial,
        iters_replan        = iters_replan,
        lns_budget          = lns_budget,
        replan_budget       = replan_budget,
        astar_limit         = astar_limit,
    )
 
 
# ════════════════════════════════════════════════════════════════════════════
#  Core LNS loop
# ════════════════════════════════════════════════════════════════════════════
 
def run_lns(
    agents, rail, paths, h_dists, max_timestep,
    iterations, neighbourhood_size,
    start_time=0, frozen_mask=None, deadline=None,
    astar_limit=1.0,
):
    if frozen_mask is None:
        frozen_mask = [False] * len(agents)
 
    best_paths = [list(p) for p in paths]
    best_delay = total_delay(agents, best_paths, max_timestep)
 
    if start_time == 0:
        plannable = [i for i in range(len(agents)) if not frozen_mask[i]]
    else:
        plannable = [i for i in range(len(agents))
                     if not frozen_mask[i]
                     and agents[i].position is not None
                     and agents[i].status not in (2, 3)]
 
    if not plannable:
        return best_paths
 
    no_improve = 0
    # Early-exit threshold: proportional to neighbourhood size, not fixed at 10
    no_improve_limit = max(15, neighbourhood_size * 4)
 
    for _ in range(iterations):
        if deadline is not None and time.time() > deadline:
            break
        if no_improve >= no_improve_limit:
            break
 
        late = [i for i in plannable
                if agent_delay(agents[i], best_paths[i], max_timestep) > 0]
        focus_id = random.choice(late) if late else random.choice(plannable)
 
        neighbourhood = find_blocking_agents(
            focus_id, best_paths, neighbourhood_size, start_time)
        neighbourhood = [i for i in neighbourhood if not frozen_mask[i]]
        if not neighbourhood:
            continue
        random.shuffle(neighbourhood)
 
        candidate_paths = [list(p) for p in best_paths]
        for nid in neighbourhood:
            candidate_paths[nid] = []
 
        success = True
        for nid in neighbourhood:
            agent = agents[nid]
            if start_time == 0:
                pos, d = agent.initial_position, agent.initial_direction
            else:
                pos = agent.position if agent.position is not None \
                      else agent.initial_position
                d   = agent.direction if agent.position is not None \
                      else agent.initial_direction
 
            constraints = [candidate_paths[i]
                           for i in range(len(agents)) if i != nid]
            new_path = space_time_astar(
                pos, d, agent.target, rail,
                constraints, max_timestep, h_dists[nid],
                start_time=start_time, time_limit=astar_limit,
            )
            if not new_path:
                success = False
                break
            candidate_paths[nid] = new_path
 
        if not success:
            no_improve += 1
            continue
 
        new_delay = total_delay(agents, candidate_paths, max_timestep)
        if new_delay <= best_delay:
            best_paths = candidate_paths
            best_delay = new_delay
            no_improve = 0
        else:
            no_improve += 1
 
    return best_paths
 
 
# ════════════════════════════════════════════════════════════════════════════
#  Module-level config store (get_path → replan)
# ════════════════════════════════════════════════════════════════════════════
_cfg: dict = {}
 
 
# ════════════════════════════════════════════════════════════════════════════
#  get_path
# ════════════════════════════════════════════════════════════════════════════

 
def get_path_worker(queue, agents, rail, max_timestep):
    try:
        result = get_path(agents, rail, max_timestep)
        queue.put(result)
    except Exception as e:
        queue.put(e)

def get_path(agents, rail, max_timestep):
    global _cfg
    n = len(agents)
    path_all = [[] for _ in range(n)]
 
    # Precompute heuristics once — reused by slack order AND A*
    h_dists = precompute_heuristics(agents, rail)
 
    # Adapt parameters to this instance
    _cfg = adapt_parameters(agents, h_dists, max_timestep)
    nbr         = _cfg['neighbourhood_size']
    astar_lim   = _cfg['astar_limit']
    lns_budget  = _cfg['lns_budget']
    iters       = _cfg['iters_initial']
 
    budget_deadline = time.time() + lns_budget
 
    # Phase 1: Prioritised Planning (slack order, uses precomputed h_dists)
    order = compute_slack_order(agents, h_dists, max_timestep)
    planned = []
    for agent_id in order:
        if time.time() > budget_deadline:
            break
        agent = agents[agent_id]
        path = space_time_astar(
            agent.initial_position, agent.initial_direction,
            agent.target, rail, planned, max_timestep,
            h_dists[agent_id], start_time=0, time_limit=astar_lim,
        )
        path_all[agent_id] = path
        planned.append(path)
 
    # Phase 2: LNS improvement
    if time.time() < budget_deadline:
        path_all = run_lns(
            agents, rail, path_all, h_dists, max_timestep,
            iterations=iters,
            neighbourhood_size=nbr,
            start_time=0,
            deadline=budget_deadline,
            astar_limit=astar_lim,
        )
 
    return path_all
 
 
# ════════════════════════════════════════════════════════════════════════════
#  replan
# ════════════════════════════════════════════════════════════════════════════
 
def replan(agents, rail, current_timestep, existing_paths, max_timestep,
           new_malfunction_agents, failed_agents):
    new_paths = [list(p) for p in existing_paths]
    h_dists = precompute_heuristics(agents, rail)

    cfg           = _cfg if _cfg else {}
    astar_lim     = cfg.get('astar_limit', 1.0)
    iters_replan  = cfg.get('iters_replan', LNS_ITERATIONS_REPLAN)
    replan_budget = cfg.get('replan_budget', LNS_TIME_BUDGET_REPLAN)
    nbr           = cfg.get('neighbourhood_size', LNS_NEIGHBOURHOOD_SIZE)

    # ── Fix prefix for ALL active agents before doing anything else ───────
    # If an agent was blocked, its actual position lags behind its planned
    # path. Truncate and pad to actual position so no path has a jump.
    for agent in agents:
        i = agent.handle
        if agent.status in (2, 3) or agent.position is None:
            continue
        actual_pos = agent.position
        prefix = list(existing_paths[i][:current_timestep])
        # Pad if path is short
        while len(prefix) < current_timestep:
            prefix.append(actual_pos)
        # Override the last entry to match actual position
        if prefix and prefix[-1] != actual_pos:
            prefix[-1] = actual_pos
        new_paths[i] = prefix  # suffix will be added below or by LNS

    replan_set = set(failed_agents) | set(new_malfunction_agents)

    # ── Fix malfunctioning/failed agents with forced waits + A* suffix ────
    for agent_id in replan_set:
        agent = agents[agent_id]
        if agent.status in (2, 3) or agent.position is None:
            continue

        cur_pos, cur_dir = agent.position, agent.direction
        mal_dur = (agent.malfunction_data.get("malfunction", 0)
                   if agent.malfunction_data else 0)

        prefix       = new_paths[agent_id][:current_timestep]  # already corrected above
        wait_segment = [cur_pos] * (mal_dur + 1)
        resume_t     = current_timestep + mal_dur

        if cur_pos == agent.target:
            new_paths[agent_id] = prefix + wait_segment
            continue

        constraints = [new_paths[i] for i in range(len(agents)) if i != agent_id]
        suffix = space_time_astar(
            cur_pos, cur_dir, agent.target, rail,
            constraints, max_timestep, h_dists[agent_id],
            start_time=resume_t, time_limit=astar_lim,
        )
        new_paths[agent_id] = prefix + wait_segment + (suffix[1:] if suffix else [])

    # ── LNS over all active agents ────────────────────────────────────────
    frozen_mask = [
        agent.status in (2, 3) or agent.position is None
        for agent in agents
    ]
    new_paths = run_lns(
        agents, rail, new_paths, h_dists, max_timestep,
        iterations=iters_replan,
        neighbourhood_size=nbr,
        start_time=current_timestep,
        frozen_mask=frozen_mask,
        deadline=time.time() + replan_budget,
        astar_limit=astar_lim,
    )

    return new_paths

 
######### Cooperative A* ###############
# # ════════════════════════════════════════════════════════════════════════════
# #  get_path — initial planning
# # ════════════════════════════════════════════════════════════════════════════
 
# def get_path(agents: List[EnvAgent], rail: GridTransitionMap,
#              max_timestep: int) -> List[List[tuple]]:
#     """
#     Plan collision-free paths for all agents.
#     Order: earliest deadline first so tight-deadline trains get priority
#     access to the shortest paths, minimising penalty exposure.
#     """
#     n = len(agents)
#     path_all = [[] for _ in range(n)]
 
#     order = sorted(
#         range(n),
#         key=lambda i: (agents[i].deadline if agents[i].deadline is not None
#                        else max_timestep)
#     )
 
#     planned: List[list] = []
 
#     for agent_id in order:
#         agent = agents[agent_id]
#         h_dist = bfs_heuristic(agent.target, rail)
 
#         path = space_time_astar(
#             agent.initial_position,
#             agent.initial_direction,
#             agent.target,
#             rail,
#             planned,
#             max_timestep,
#             h_dist,
#             start_time=0,
#         )
 
#         path_all[agent_id] = path
#         planned.append(path)
 
#     return path_all
 
 
# # ════════════════════════════════════════════════════════════════════════════
# #  replan — malfunction / collision recovery
# # ════════════════════════════════════════════════════════════════════════════
 
# def replan(
#     agents: List[EnvAgent],
#     rail: GridTransitionMap,
#     current_timestep: int,
#     existing_paths: List[List[tuple]],
#     max_timestep: int,
#     new_malfunction_agents: List[int],
#     failed_agents: List[int],
# ) -> List[List[tuple]]:
#     """
#     Repair paths after a malfunction or collision.
 
#     Critical invariant
#     ------------------
#     path_controller calls get_action(agent_id, path[t], env) whenever
#     t < len(path) AND status != 3.  get_action reads agent.position, which
#     is None for any agent that has never been spawned (status == 0).
#     There is no way to spawn such an agent after t=0, so we must NOT extend
#     their path — doing so would cause a TypeError on agent.position[0].
 
#     Therefore: skip any agent whose position is None (status 0, not spawned)
#     or who has already finished (status 2 / 3).
 
#     Path layout for a replanned active agent
#     -----------------------------------------
#     new_path = existing_path[:current_timestep]    <- immutable history
#              + [cur_pos] * (mal_dur + 1)           <- sit still during fault
#              + suffix[1:]                          <- A* route to goal
#                                                       (suffix[0] == cur_pos)
#     Indices:  0 .. t-1  |  t .. t+mal_dur  |  t+mal_dur+1 ..
#     """
#     new_paths = [list(p) for p in existing_paths]
 
#     replan_set = set(failed_agents) | set(new_malfunction_agents)
 
#     # Earliest deadline first so tight-deadline agents get priority
#     replan_order = sorted(
#         replan_set,
#         key=lambda i: (agents[i].deadline if agents[i].deadline is not None
#                        else max_timestep)
#     )
 
#     for agent_id in replan_order:
#         agent = agents[agent_id]
 
#         # ── Skip agents we cannot or need not replan ─────────────────────
#         # status 0 : not yet spawned — position is None, cannot be moved
#         # status 2 : at goal (should be 3 with remove_agents_at_target=True)
#         # status 3 : done and removed
#         # position None : same as status 0 guard, defensive check
#         if agent.status in (2, 3):
#             continue
#         if agent.position is None:
#             # Cannot replan an unspawned agent — leave path as-is so the
#             # path_controller keeps sending NOTHING (len(path) <= t).
#             continue
 
#         cur_pos = agent.position
#         cur_dir = agent.direction
#         mal_dur = (agent.malfunction_data.get("malfunction", 0)
#                    if agent.malfunction_data else 0)
 
#         # Already at goal — just extend with waits so path doesn't expire
#         if cur_pos == agent.target:
#             prefix = list(existing_paths[agent_id][:current_timestep])
#             new_paths[agent_id] = prefix + [cur_pos] * (mal_dur + 1)
#             continue
 
#         # ── Build immutable history prefix ───────────────────────────────
#         # Keep exactly current_timestep entries so path[current_timestep]
#         # maps to the actual current position.
#         raw_prefix = list(existing_paths[agent_id][:current_timestep])
#         # If existing path was shorter than current_timestep (e.g. agent
#         # had an empty path), pad with cur_pos so index alignment is right.
#         if len(raw_prefix) < current_timestep:
#             raw_prefix += [cur_pos] * (current_timestep - len(raw_prefix))
#         prefix = raw_prefix
 
#         # ── Forced wait segment (malfunction period) ─────────────────────
#         # Covers absolute timesteps current_timestep .. current_timestep+mal_dur
#         wait_segment = [cur_pos] * (mal_dur + 1)
 
#         # ── Resume A* after malfunction clears ───────────────────────────
#         resume_t = current_timestep + mal_dur
 
#         # Build constraints from all OTHER agents' already-updated paths
#         constraints = [new_paths[i] for i in range(len(agents))
#                        if i != agent_id]
 
#         h_dist = bfs_heuristic(agent.target, rail)
 
#         suffix = space_time_astar(
#             cur_pos, cur_dir, agent.target,
#             rail, constraints, max_timestep, h_dist,
#             start_time=resume_t,
#         )
 
#         if suffix:
#             # suffix[0] == cur_pos at resume_t, already covered by
#             # wait_segment[-1], so drop it to avoid duplication.
#             new_paths[agent_id] = prefix + wait_segment + suffix[1:]
#         else:
#             # No path found — keep agent waiting; evaluator will penalise
#             new_paths[agent_id] = prefix + wait_segment
 
#     return new_paths


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
