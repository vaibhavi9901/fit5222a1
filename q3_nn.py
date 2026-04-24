
from lib_piglet.utils.tools import eprint
from typing import List, Tuple, Dict, Optional
import glob, os, sys,time,json, heapq, random
from collections import deque
import multiprocessing as mp
import numpy as np

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
test_single_instance = False
level = 1
test = 1
debug = False
visualizer = False


# ── Neural-network priority module ──────────────────────────────────────────
_NN_AVAILABLE = False
_priority_net = None
try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _script_dir)
    from nn_priority import AgentPriorityNet, extract_features as _extract_nn_features
    _priority_net = AgentPriorityNet()
    _model_path = os.path.join(_script_dir, "model_weights.npz")
    if _priority_net.load(_model_path):
        _NN_AVAILABLE = True
    else:
        eprint("[solution] model_weights.npz not found — falling back to 4-seed baseline.")
except Exception as e:
    eprint(f"[solution] NN module unavailable ({e}). Using 4-seed baseline.")
 
# ─────────────────────────────────────────────────────────────────────────────
#  BFS heuristic cache
# ─────────────────────────────────────────────────────────────────────────────
_heuristic_cache: Dict[tuple, dict] = {}
_rail_ref = None
 
 
def get_or_compute_heuristic(goal: tuple, rail: GridTransitionMap) -> dict:
    if goal not in _heuristic_cache:
        _heuristic_cache[goal] = bfs_heuristic(goal, rail)
    return _heuristic_cache[goal]
 
 
def bfs_heuristic(goal: tuple, rail: GridTransitionMap) -> dict:
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
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  Conflict detection
# ─────────────────────────────────────────────────────────────────────────────
def has_conflict(new_loc: tuple, cur_loc: tuple, t: int,
                 constraint_paths: list) -> bool:
    for p in constraint_paths:
        if not p:
            continue
        if t + 1 < len(p):
            if p[t + 1] == new_loc:
                return True
            if p[t + 1] == cur_loc and p[t] == new_loc:
                return True
        else:
            if p[-1] == new_loc:
                return True
    return False
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  Space-Time A*
# ─────────────────────────────────────────────────────────────────────────────
def space_time_astar(
    start: tuple,
    start_dir: int,
    goal: tuple,
    rail: GridTransitionMap,
    constraint_paths: list,
    max_timestep: int,
    h_dist: dict,
    start_time: int = 0,
    time_limit: float = None,
    deadline: int = None,
    deadline_weight: float = 2.0,
) -> list:
    if start not in h_dist:
        return []
    search_start_wall = time.time()
    expansion_count = 0
    start_state = (start[0], start[1], start_dir, start_time)
    open_heap = [(h_dist[start], 0, start[0], start[1], start_dir, start_time)]
    visited = set()
    best_g: Dict[tuple, int] = {start_state: 0}
    parent: Dict[tuple, Optional[tuple]] = {start_state: None}
 
    while open_heap:
        expansion_count += 1
        if time_limit is not None and expansion_count % 500 == 0:
            if time.time() - search_start_wall > time_limit:
                return []
 
        f, g, x, y, direction, t = heapq.heappop(open_heap)
        state = (x, y, direction, t)
        if state in visited:
            continue
        visited.add(state)
 
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
 
        def try_push(nx, ny, new_dir, new_t, new_g):
            ns = (nx, ny, new_dir, new_t)
            if ns in visited:
                return
            if ns not in best_g or new_g < best_g[ns]:
                best_g[ns] = new_g
                parent[ns] = state
                new_h = h_dist.get((nx, ny))
                if new_h is None:
                    return
                delay = max(0, new_t - deadline) if deadline is not None else 0
                heapq.heappush(open_heap,
                               (new_g + new_h + deadline_weight * delay,
                                new_g, nx, ny, new_dir, new_t))
 
        for action, valid in enumerate(rail.get_transitions(x, y, direction)):
            if not valid:
                continue
            nx, ny = x, y
            if   action == Directions.NORTH: nx -= 1
            elif action == Directions.EAST:  ny += 1
            elif action == Directions.SOUTH: nx += 1
            elif action == Directions.WEST:  ny -= 1
            if has_conflict((nx, ny), cur_loc, t, constraint_paths):
                continue
            if (nx, ny) not in h_dist:
                continue
            try_push(nx, ny, action, t + 1, g + 1)
 
        if not has_conflict(cur_loc, cur_loc, t, constraint_paths):
            try_push(x, y, direction, t + 1, g + 1)
 
    return []
 
 
def _astar_with_fallback(start, start_dir, goal, rail, constraints,
                         horizon, h_dist, start_time, deadline, time_limit):
    path = space_time_astar(start, start_dir, goal, rail, constraints,
                            horizon, h_dist, start_time=start_time,
                            deadline=deadline, time_limit=time_limit)
    if path:
        return path
    path = space_time_astar(start, start_dir, goal, rail, [],
                            horizon, h_dist, start_time=start_time,
                            deadline=deadline, time_limit=time_limit)
    return path
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  Parallel worker
#  Args tuple is always 6 elements:
#    (agents, rail, max_timestep, seed, nn_features, precomputed_h)
#  nn_features and precomputed_h may be None.
# ─────────────────────────────────────────────────────────────────────────────
def _plan_worker(args):
    agents, rail, max_timestep, seed, nn_features, precomputed_h = args
 
    # Seed local BFS cache from precomputed maps (avoids redundant BFS calls).
    # precomputed_h maps goal_tuple → dist_dict.
    local_cache: Dict[tuple, dict] = dict(precomputed_h) if precomputed_h else {}
 
    def local_h(goal):
        if goal not in local_cache:
            local_cache[goal] = bfs_heuristic(goal, rail)
        return local_cache[goal]
 
    n = len(agents)
    path_all = [[] for _ in range(n)]
 
    def h(i):
        return local_h(agents[i].target).get(agents[i].initial_position,
                                              max_timestep)
 
    def ddl(i):
        return (agents[i].deadline if agents[i].deadline is not None
                else max_timestep)
 
    if seed == 0:
        order = sorted(range(n), key=lambda i: (ddl(i), h(i)))
    elif seed == 1:
        order = sorted(range(n), key=lambda i: ddl(i) - h(i))
    elif seed == 2:
        order = sorted(range(n), key=lambda i: -h(i))
    elif seed == 3:
        order = list(range(n))
        random.seed(42)
        random.shuffle(order)
    elif seed == 4 and nn_features is not None and _priority_net is not None:
        try:
            scores = _priority_net.forward(nn_features)
            order  = np.argsort(scores).tolist()
        except Exception:
            order = sorted(range(n), key=lambda i: (ddl(i), h(i)))
    else:
        order = sorted(range(n), key=lambda i: (ddl(i), h(i)))
 
    per_agent_limit = max(2.0, min(15.0, 60.0 / max(1, n)))
    worker_start    = time.time()
 
    planned: List[list] = []
    for agent_id in order:
        agent = agents[agent_id]
        if agent.initial_position is None or agent.target is None:
            planned.append([])
            continue
 
        elapsed   = time.time() - worker_start
        remaining = len(order) - len(planned)
        time_left = max(1.0, per_agent_limit * remaining - elapsed)
 
        h_dist = local_h(agent.target)
        path = _astar_with_fallback(
            agent.initial_position, agent.initial_direction,
            agent.target, rail, planned, max_timestep, h_dist,
            start_time=0, deadline=agent.deadline,
            time_limit=min(per_agent_limit, time_left),
        )
        path_all[agent_id] = path
        planned.append(path)
 
    failed = sum(1 for p in path_all if not p)
    cost   = sum(len(p) - 1 for p in path_all if p)
    return path_all, (failed, cost)
 
 
# ════════════════════════════════════════════════════════════════════════════
#  get_path — initial planning
# ════════════════════════════════════════════════════════════════════════════
_MP_CTX = 'fork' if sys.platform.startswith('linux') else 'spawn'
 
def get_path(agents: List[EnvAgent], rail: GridTransitionMap,
             max_timestep: int) -> List[List[tuple]]:
    global _rail_ref
    _heuristic_cache.clear()
    _rail_ref = rail
 
    # Pre-compute BFS maps once — shared with all workers via fork / passed explicitly.
    nn_features  = None
    h_dists_main: Dict[int, dict] = {}
    # precomputed_h maps goal_tuple → dist_dict (what workers need for local_cache)
    precomputed_h: Dict[tuple, dict] = {}
 
    for i, ag in enumerate(agents):
        if ag.target is not None and ag.initial_position is not None:
            if ag.target not in precomputed_h:
                precomputed_h[ag.target] = bfs_heuristic(ag.target, rail)
                _heuristic_cache[ag.target] = precomputed_h[ag.target]
            h_dists_main[i] = precomputed_h[ag.target]
 
    if _NN_AVAILABLE and _priority_net is not None:
        try:
            grid_rows = getattr(rail, 'height', 50) or 50
            grid_cols = getattr(rail, 'width',  50) or 50
            _agent_deadlines = [
                ag.deadline if ag.deadline is not None else max_timestep
                for ag in agents
            ]
            nn_features = _extract_nn_features(
                agents, h_dists_main, max_timestep, grid_rows, grid_cols,
                deadlines=_agent_deadlines,
            )
        except Exception as e:
            eprint(f"[solution] NN feature extraction failed ({e})")
 
    NUM_WORKERS = min(4, mp.cpu_count())
 
    # All args tuples are always 6 elements — no more unpacking mismatches.
    base_args = [
        (agents, rail, max_timestep, seed, None, precomputed_h)
        for seed in range(NUM_WORKERS)
    ]
    nn_args = (
        [(agents, rail, max_timestep, 4, nn_features, precomputed_h)]
        if nn_features is not None and _NN_AVAILABLE and _priority_net is not None
        else []
    )
    all_args = base_args + nn_args
 
    try:
        ctx = mp.get_context(_MP_CTX)
        with ctx.Pool(processes=min(len(all_args), mp.cpu_count())) as pool:
            results = pool.map(_plan_worker, all_args)
        best_paths, best_score = min(results, key=lambda r: r[1])
    except Exception as e:
        eprint(f"Parallel planning failed ({e}), falling back to serial")
        # Serial fallback also uses 6-element tuple
        best_paths, best_score = _plan_worker(
            (agents, rail, max_timestep, 0, None, precomputed_h)
        )
 
    eprint(f"[get_path] best=(failed={best_score[0]}, cost={best_score[1]})")
    return best_paths
 
 
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
    new_paths  = [list(p) for p in existing_paths]
    replan_set = set(failed_agents) | set(new_malfunction_agents)
    n          = len(agents)
 
    remaining_steps = max(20, max_timestep - current_timestep)
 
    if n >= 75:
        search_horizon    = current_timestep + min(remaining_steps, 80)
        replan_time_limit = 0.5
    elif n >= 25:
        search_horizon    = current_timestep + min(remaining_steps, 150)
        replan_time_limit = max(0.3, 10.0 / n)
    else:
        search_horizon    = current_timestep + remaining_steps
        replan_time_limit = 0.25
 
    def deadline_key(i):
        return agents[i].deadline if agents[i].deadline is not None else max_timestep
 
    def _replan_one(agent_id: int):
        agent = agents[agent_id]
        if agent.status in (2, 3) or agent.position is None:
            return
 
        cur_pos = agent.position
        cur_dir = agent.direction
        mal_dur = (agent.malfunction_data.get("malfunction", 0)
                   if agent.malfunction_data else 0)
 
        if cur_pos == agent.target:
            prefix = list(existing_paths[agent_id][:current_timestep])
            new_paths[agent_id] = prefix + [cur_pos] * max(mal_dur, 1)
            return
 
        raw_prefix = list(existing_paths[agent_id][:current_timestep])
        if len(raw_prefix) < current_timestep:
            raw_prefix += [cur_pos] * (current_timestep - len(raw_prefix))
 
        wait_segment = [cur_pos] * mal_dur
        resume_t     = current_timestep + mal_dur
 
        constraints = [new_paths[i] for i in range(n) if i != agent_id]
        h_dist      = get_or_compute_heuristic(agent.target, rail)
 
        if cur_pos not in h_dist:
            new_paths[agent_id] = raw_prefix + wait_segment
            return
 
        suffix = _astar_with_fallback(
            cur_pos, cur_dir, agent.target, rail, constraints,
            search_horizon, h_dist,
            start_time=resume_t,
            deadline=agent.deadline,
            time_limit=replan_time_limit,
        )
 
        if suffix:
            tail = suffix[1:] if mal_dur > 0 else suffix
            new_paths[agent_id] = raw_prefix + wait_segment + tail
        else:
            new_paths[agent_id] = raw_prefix + wait_segment
 
    def _find_cascade_conflicts(already_replanned: set) -> set:
        affected  = set()
        check_end = min(current_timestep + 40, max_timestep)
        for agent_id in range(n):
            if agent_id in already_replanned:
                continue
            agent = agents[agent_id]
            if agent.status in (2, 3) or agent.position is None:
                continue
            path        = new_paths[agent_id]
            other_paths = [new_paths[i] for i in range(n) if i != agent_id]
            for t in range(current_timestep, min(len(path) - 1, check_end)):
                if has_conflict(path[t + 1], path[t], t, other_paths):
                    affected.add(agent_id)
                    break
        return affected
 
    for agent_id in sorted(replan_set, key=deadline_key):
        _replan_one(agent_id)
 
    for _ in range(3):
        affected = _find_cascade_conflicts(replan_set)
        if not affected:
            break
        for agent_id in sorted(affected, key=deadline_key):
            _replan_one(agent_id)
        replan_set |= affected
 
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