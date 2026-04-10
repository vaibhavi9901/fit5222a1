
from lib_piglet.utils.tools import eprint
from typing import List, Tuple, Dict, Optional
import glob, os, sys,time,json, heapq, random
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

############ LNS-based planning ###############

# ── LNS parameters ───────────────────────────────────────────────────────────
LNS_ITERATIONS_INITIAL  = 200   # iterations in get_path
#LNS_NEIGHBOURHOOD_SIZE  = 5     # agents per LNS neighbourhood
LNS_ITERATIONS_REPLAN   = 50    # iterations per replan call
LNS_TIME_BUDGET         = 60.0  # seconds budget for LNS in get_path

# ════════════════════════════════════════════════════════════════════════════
#  Slack-based priority ordering   (Chen et al. §"Slack Based Priority")
# ════════════════════════════════════════════════════════════════════════════
 
def compute_slack_order(agents: List[EnvAgent], rail: GridTransitionMap,
                        max_timestep: int) -> List[int]:
    """
    slack_i = deadline_i - earliest_possible_arrival_i
    Lower slack → tighter schedule → higher priority (planned first).
    Ties broken by BFS distance ascending (shorter trips first).
 
    Returns list of agent indices in planning order.
    """
    slacks = []
    for i, agent in enumerate(agents):
        h = bfs_heuristic(agent.target, rail)
        earliest = h.get(agent.initial_position, max_timestep)
        ddl = agent.deadline if agent.deadline is not None else max_timestep
        slack = ddl - earliest
        slacks.append((slack, earliest, i))
    slacks.sort()                        # ascending slack → tightest first
    return [i for _, _, i in slacks]
 
 
# ════════════════════════════════════════════════════════════════════════════
#  Delay metric   (Chen et al. objective: minimise total arrival delay)
# ════════════════════════════════════════════════════════════════════════════
 
def arrival_time(path: list) -> int:
    """Index of last element = arrival timestep."""
    return len(path) - 1 if path else 0
 
 
def agent_delay(agent: EnvAgent, path: list, max_timestep: int) -> int:
    """
    Delay for one agent: max(0, arrival_time - deadline).
    If no path, penalise with max_timestep.
    """
    if not path:
        return max_timestep
    ddl = agent.deadline if agent.deadline is not None else max_timestep
    arr = arrival_time(path)
    return max(0, arr - ddl)
 
 
def total_delay(agents: List[EnvAgent], paths: List[list],
                max_timestep: int) -> int:
    return sum(agent_delay(agents[i], paths[i], max_timestep)
               for i in range(len(agents)))
 
 
# ════════════════════════════════════════════════════════════════════════════
#  LNS neighbourhood selection   (Chen et al. §"Delay-based neighbourhood")
# ════════════════════════════════════════════════════════════════════════════
 
def find_blocking_agents(focus_id: int, paths: List[list],
                         neighbourhood_size: int,
                         start_time: int = 0) -> List[int]:
    """
    Find up to neighbourhood_size-1 agents whose paths spatially overlap
    with the focus agent's path (potential blockers). Returns ids including
    focus_id.
    """
    focus_path = paths[focus_id]
    if not focus_path:
        return [focus_id]
 
    # Build a set of (cell, t) occupied by focus agent
    focus_cells = set()
    for t, cell in enumerate(focus_path):
        if t >= start_time:
            focus_cells.add(cell)
 
    overlap_count: Dict[int, int] = {}
    for other_id, p in enumerate(paths):
        if other_id == focus_id or not p:
            continue
        for t, cell in enumerate(p):
            if t >= start_time and cell in focus_cells:
                overlap_count[other_id] = overlap_count.get(other_id, 0) + 1
 
    # Sort by overlap descending, take top neighbourhood_size-1
    ranked = sorted(overlap_count.items(), key=lambda x: -x[1])
    neighbours = [focus_id] + [aid for aid, _ in ranked[:neighbourhood_size - 1]]
    return neighbours
 
 
def precompute_heuristics(agents: List[EnvAgent],
                          rail: GridTransitionMap) -> List[dict]:
    """Precompute BFS heuristic for every agent (cached per goal cell)."""
    cache: Dict[tuple, dict] = {}
    result = []
    for agent in agents:
        if agent.target not in cache:
            cache[agent.target] = bfs_heuristic(agent.target, rail)
        result.append(cache[agent.target])
    return result
 
 
# ════════════════════════════════════════════════════════════════════════════
#  Core LNS loop
# ════════════════════════════════════════════════════════════════════════════
 
def run_lns(
    agents: List[EnvAgent],
    rail: GridTransitionMap,
    paths: List[list],
    h_dists: List[dict],
    max_timestep: int,
    iterations: int,
    neighbourhood_size: int,
    start_time: int = 0,
    frozen_mask: Optional[List[bool]] = None,
    deadline: float = None,
) -> List[list]:
    """
    MAPF-LNS with delay-based neighbourhood selection.
 
    frozen_mask[i] = True means agent i's path cannot be changed
    (used during replanning for already-finished agents).
 
    In each iteration:
      1. Pick a random agent that is late (delay > 0). If none late, pick
         any non-frozen agent randomly (keeps diversity).
      2. Collect neighbourhood (focus + blockers).
      3. Remove their paths from the constraint set.
      4. Replan in random priority order using Space-Time A*.
      5. Accept if total delay improves (or stays equal with shorter paths).
    """
    if frozen_mask is None:
        frozen_mask = [False] * len(agents)
 
    best_paths = [list(p) for p in paths]
    best_delay = total_delay(agents, best_paths, max_timestep)
 
    # Identify plannable agents
    plannable = [i for i in range(len(agents))
                 if not frozen_mask[i] and agents[i].status not in (2, 3)
                 and agents[i].position is not None or start_time == 0]
    # At initial planning time all agents are plannable
    if start_time == 0:
        plannable = [i for i in range(len(agents)) if not frozen_mask[i]]
 
    if not plannable:
        return best_paths
 
    for _ in range(iterations):
        if deadline is not None and time.time() > deadline:
            break
        # ── Select focus agent ────────────────────────────────────────
        late_agents = [i for i in plannable
                       if agent_delay(agents[i], best_paths[i], max_timestep) > 0]
        focus_id = random.choice(late_agents) if late_agents \
                   else random.choice(plannable)
 
        # ── Build neighbourhood ───────────────────────────────────────
        neighbourhood = find_blocking_agents(
            focus_id, best_paths, neighbourhood_size, start_time)
        neighbourhood = [i for i in neighbourhood if not frozen_mask[i]]
        if not neighbourhood:
            continue
        random.shuffle(neighbourhood)
 
        # ── Temporarily remove neighbourhood paths ────────────────────
        candidate_paths = [list(p) for p in best_paths]
        for nid in neighbourhood:
            candidate_paths[nid] = []
 
        # ── Replan neighbourhood in shuffled order ────────────────────
        success = True
        for nid in neighbourhood:
            agent = agents[nid]
            # Determine start position and direction for this agent
            if start_time == 0:
                pos = agent.initial_position
                d   = agent.initial_direction
            else:
                pos = agent.position if agent.position is not None \
                      else agent.initial_position
                d   = agent.direction if agent.position is not None \
                      else agent.initial_direction
 
            # Build constraints from all paths except this neighbourhood
            constraints = [candidate_paths[i]
                           for i in range(len(agents)) if i != nid]
 
            new_path = space_time_astar(
                pos, d, agent.target, rail,
                constraints, max_timestep, h_dists[nid],
                start_time=start_time,
            )
            if not new_path:
                success = False
                break
            candidate_paths[nid] = new_path
 
        if not success:
            continue
 
        # ── Accept if delay improves ──────────────────────────────────
        new_delay = total_delay(agents, candidate_paths, max_timestep)
        if new_delay <= best_delay:
            best_paths = candidate_paths
            best_delay = new_delay
 
    return best_paths
 
 
# ════════════════════════════════════════════════════════════════════════════
#  get_path
# ════════════════════════════════════════════════════════════════════════════
 
def get_path(agents: List[EnvAgent], rail: GridTransitionMap,
             max_timestep: int) -> List[List[tuple]]:
    """
    1. Slack-based priority ordering.
    2. Space-Time A* in priority order (Prioritised Planning).
    3. Delay-based LNS improvement.
    """
    n = len(agents)
    path_all = [[] for _ in range(n)]
 
    # ── Dynamic neighbourhood size: one neighbourhood per agent ──────────
    neighbourhood_size = n
 
    # ── Precompute heuristics ─────────────────────────────────────────────
    h_dists = precompute_heuristics(agents, rail)
 
    # ── Phase 1: Prioritised Planning with slack ordering ─────────────────
    order = compute_slack_order(agents, rail, max_timestep)
    planned: List[list] = []   # grows as constraints
 
    for agent_id in order:
        agent = agents[agent_id]
        path = space_time_astar(
            agent.initial_position, agent.initial_direction,
            agent.target, rail, planned, max_timestep,
            h_dists[agent_id], start_time=0,
        )
        path_all[agent_id] = path
        planned.append(path)
 
    # ── Phase 2: LNS improvement ──────────────────────────────────────────
    path_all = run_lns(
        agents, rail, path_all, h_dists, max_timestep,
        iterations=LNS_ITERATIONS_INITIAL,
        neighbourhood_size=neighbourhood_size,
        start_time=0,
        deadline=time.time() + LNS_TIME_BUDGET,
    )
 
    return path_all
 
 
# ════════════════════════════════════════════════════════════════════════════
#  replan  — LNS-based periodic replanning
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
    LNS-based replanning (Chen et al. §"Partial Replanning using LNS").
 
    Steps:
      1. Copy paths; keep history up to current_timestep immutable.
      2. Force malfunctioning agents to wait at their current cell for the
         malfunction duration (they physically cannot move).
      3. Run LNS over all active agents, guided by total delay objective,
         for LNS_ITERATIONS_REPLAN iterations.
      4. Frozen mask: skip agents that are done or unspawned.
    """
    new_paths = [list(p) for p in existing_paths]
    h_dists = precompute_heuristics(agents, rail)
 
    replan_set = set(failed_agents) | set(new_malfunction_agents)
 
    # ── Step 1+2: Fix malfunctioning agents with forced waits ─────────────
    for agent_id in replan_set:
        agent = agents[agent_id]
        if agent.status in (2, 3) or agent.position is None:
            continue
 
        cur_pos = agent.position
        cur_dir = agent.direction
        mal_dur = (agent.malfunction_data.get("malfunction", 0)
                   if agent.malfunction_data else 0)
 
        prefix = list(existing_paths[agent_id][:current_timestep])
        if len(prefix) < current_timestep:
            prefix += [cur_pos] * (current_timestep - len(prefix))
 
        wait_segment = [cur_pos] * (mal_dur + 1)
        resume_t = current_timestep + mal_dur
 
        if cur_pos == agent.target:
            new_paths[agent_id] = prefix + wait_segment
            continue
 
        constraints = [new_paths[i] for i in range(len(agents))
                       if i != agent_id]
 
        suffix = space_time_astar(
            cur_pos, cur_dir, agent.target, rail,
            constraints, max_timestep, h_dists[agent_id],
            start_time=resume_t,
        )
 
        new_paths[agent_id] = prefix + wait_segment + (suffix[1:] if suffix else [])
 
    # ── Step 3: LNS over all active agents ───────────────────────────────
    # Frozen: done agents, unspawned agents (position is None after t=0)
    frozen_mask = [
        agent.status in (2, 3) or agent.position is None
        for agent in agents
    ]
 
    new_paths = run_lns(
        agents, rail, new_paths, h_dists, max_timestep,
        iterations=LNS_ITERATIONS_REPLAN,
        #neighbourhood_size=LNS_NEIGHBOURHOOD_SIZE,
        neighbourhood_size = len(agents),
        start_time=current_timestep,
        frozen_mask=frozen_mask,
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
