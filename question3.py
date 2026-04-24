
from lib_piglet.utils.tools import eprint
from typing import List, Tuple, Dict, Optional,  Set
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
test_single_instance = True
level = 4
test = 0
debug = True
visualizer = False

    # ═════════════════════════════════════════════════════════════════════════════
#  BFS heuristic cache
# ═════════════════════════════════════════════════════════════════════════════
_heuristic_cache: Dict[tuple, dict] = {}
 
 
def get_or_compute_heuristic(goal: tuple, rail: GridTransitionMap) -> dict:
    if goal not in _heuristic_cache:
        _heuristic_cache[goal] = bfs_heuristic(goal, rail)
    return _heuristic_cache[goal]
 
 
def bfs_heuristic(goal: tuple, rail: GridTransitionMap) -> dict:
    """Backward BFS → {(r,c): min_steps_to_goal}."""
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
 
 
# ═════════════════════════════════════════════════════════════════════════════
#  Conflict detection
# ═════════════════════════════════════════════════════════════════════════════
 
def has_conflict(new_loc: tuple, cur_loc: tuple, t: int,
                 constraint_paths: list) -> bool:
    """Vertex + edge (swap) conflict check."""
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
 
 
def find_first_conflict(paths: List[List[tuple]]) -> Optional[tuple]:
    """
    Return (agent_i, agent_j, timestep, conflict_type) for the first
    conflict found, or None if no conflict exists.
    conflict_type: 'vertex' or 'edge'
    """
    max_t = max((len(p) for p in paths if p), default=0)
    n = len(paths)
    for t in range(max_t):
        locs: Dict[tuple, int] = {}
        for i, p in enumerate(paths):
            if not p:
                continue
            loc = p[t] if t < len(p) else p[-1]
            if loc in locs:
                return (locs[loc], i, t, 'vertex')
            locs[loc] = i
        # edge conflicts
        for i in range(n):
            if not paths[i]:
                continue
            for j in range(i + 1, n):
                if not paths[j]:
                    continue
                li = paths[i][t] if t < len(paths[i]) else paths[i][-1]
                lj = paths[j][t] if t < len(paths[j]) else paths[j][-1]
                li1 = paths[i][t + 1] if t + 1 < len(paths[i]) else paths[i][-1]
                lj1 = paths[j][t + 1] if t + 1 < len(paths[j]) else paths[j][-1]
                if li == lj1 and lj == li1:
                    return (i, j, t, 'edge')
    return None
 
 
# ═════════════════════════════════════════════════════════════════════════════
#  Space-Time A*  (low-level planner used by every component)
# ═════════════════════════════════════════════════════════════════════════════
 
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
    extra_vertex_constraints: Set[tuple] = None,   # {(t, r, c)}
    extra_edge_constraints: Set[tuple] = None,     # {(t, r1,c1, r2,c2)}
) -> list:
    """
    STA* with:
      • deadline-aware cost: f = g + h + 2*delay_past_deadline
      • hard vertex / edge constraints (for symmetry breaking)
      • timeout → wait-in-place fallback
    """
    if start not in h_dist:
        return []
 
    wall0 = time.time()
    expansions = 0
 
    start_state = (start[0], start[1], start_dir, start_time)
    open_heap = [(h_dist[start], 0, start[0], start[1], start_dir, start_time)]
    visited: Set[tuple] = set()
    best_g: Dict[tuple, int] = {start_state: 0}
    parent: Dict[tuple, Optional[tuple]] = {start_state: None}
 
    evc = extra_vertex_constraints or set()
    eec = extra_edge_constraints or set()
 
    while open_heap:
        expansions += 1
        if time_limit is not None and expansions % 500 == 0:
            if time.time() - wall0 > time_limit:
                return [start] * (max_timestep - start_time + 1)
 
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
            # hard vertex constraint
            if (new_t, nx, ny) in evc:
                return
            if ns not in best_g or new_g < best_g[ns]:
                best_g[ns] = new_g
                parent[ns] = state
                nh = h_dist.get((nx, ny))
                if nh is None:
                    return
                delay = max(0, new_t - deadline) if deadline is not None else 0
                heapq.heappush(open_heap,
                               (new_g + nh + 2 * delay,
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
            if (t, x, y, nx, ny) in eec:   # hard edge constraint
                continue
            if (nx, ny) not in h_dist:
                continue
            try_push(nx, ny, action, t + 1, g + 1)
 
        # wait
        if not has_conflict(cur_loc, cur_loc, t, constraint_paths):
            if (t, x, y, x, y) not in eec:
                try_push(x, y, direction, t + 1, g + 1)
 
    return []
 
 
# ═════════════════════════════════════════════════════════════════════════════
#  PIBT  (Push-based Individual with Back-tracking)
# ═════════════════════════════════════════════════════════════════════════════
 
def _pibt_step(agent_id: int,
               positions: Dict[int, tuple],
               directions: Dict[int, int],
               goals: Dict[int, tuple],
               h_dists: Dict[int, dict],
               rail: GridTransitionMap,
               reserved: Dict[tuple, int],
               visited_ids: Set[int],
               max_recurse: int = 0) -> bool:
    """
    Try to move agent_id one step closer to its goal.
    Recursively resolves blocking agents (priority inheritance).
    Returns True if the agent successfully moved (or is already at goal).
    """
    if agent_id in visited_ids:
        return False
    visited_ids.add(agent_id)
 
    pos = positions[agent_id]
    goal = goals[agent_id]
    if pos == goal:
        reserved[pos] = agent_id
        return True
 
    d = directions[agent_id]
    h_dist = h_dists[agent_id]
 
    # Enumerate candidate moves, sorted by heuristic
    candidates = []
    for action, valid in enumerate(rail.get_transitions(pos[0], pos[1], d)):
        if not valid:
            continue
        nx, ny = pos
        if   action == Directions.NORTH: nx -= 1
        elif action == Directions.EAST:  ny += 1
        elif action == Directions.SOUTH: nx += 1
        elif action == Directions.WEST:  ny -= 1
        nh = h_dist.get((nx, ny), 1e9)
        candidates.append((nh, action, (nx, ny)))
    # Also consider waiting (last resort)
    candidates.sort()
 
    for _, action, new_pos in candidates:
        if new_pos in reserved and reserved[new_pos] != agent_id:
            blocker = reserved[new_pos]
            if max_recurse > 0:
                success = _pibt_step(blocker, positions, directions, goals,
                                     h_dists, rail, reserved, visited_ids,
                                     max_recurse - 1)
                if not success:
                    continue
            else:
                continue
        # Move is feasible
        reserved[new_pos] = agent_id
        positions[agent_id] = new_pos
        directions[agent_id] = action
        return True
 
    # Wait in place
    reserved[pos] = agent_id
    return False
 
 
def pibt_plan(agents: List[EnvAgent], rail: GridTransitionMap,
              max_timestep: int,
              h_dists: Dict[int, dict]) -> List[List[tuple]]:
    """
    Run PIBT for max_timestep steps and return complete paths.
    Priority = urgency (deadline - h_dist) ascending (tightest first).
    """
    n = len(agents)
    positions = {i: agents[i].initial_position for i in range(n)
                 if agents[i].initial_position is not None}
    directions = {i: agents[i].initial_direction for i in range(n)
                  if agents[i].initial_position is not None}
    goals = {i: agents[i].target for i in range(n)}
    paths = {i: [positions[i]] for i in positions}
 
    for t in range(max_timestep):
        # Sort by tightest deadline – h_dist (most urgent first)
        def priority(i):
            ddl = agents[i].deadline if agents[i].deadline is not None else max_timestep
            h = h_dists[i].get(positions.get(i, goals[i]), max_timestep)
            return ddl - h  # lower = more urgent
 
        order = sorted(positions.keys(), key=priority)
        reserved: Dict[tuple, int] = {}
        visited_ids: Set[int] = set()
 
        for agent_id in order:
            if positions[agent_id] == goals[agent_id]:
                reserved[positions[agent_id]] = agent_id
                continue
            _pibt_step(agent_id, positions, directions, goals,
                       h_dists, rail, reserved, visited_ids, max_recurse=3)
 
        for i in positions:
            paths[i].append(positions[i])
 
        # Remove agents that have reached goal (they stay put)
        done = {i for i in positions if positions[i] == goals[i]}
 
    result = [[] for _ in range(n)]
    for i in range(n):
        result[i] = paths.get(i, [])
    return result
 
 
# ═════════════════════════════════════════════════════════════════════════════
#  Symmetry-breaking constraint generators
# ═════════════════════════════════════════════════════════════════════════════
 
def _detect_rectangle_conflict(path_i, path_j, t):
    """
    Returns (barrier_constraints_i, barrier_constraints_j) as sets of
    (timestep, r, c) vertex constraints.  Simplified: if two agents
    approach a bounding-rectangle from opposite corners, block the
    'later' one for the width of the rectangle.
    """
    # Lightweight version: detect if agents swap across a 1×k or k×1 block
    # (full rectangle symmetry would need more geometry)
    if t + 2 >= len(path_i) or t + 2 >= len(path_j):
        return set(), set()
 
    ri, ci = path_i[t]
    rj, cj = path_j[t]
    ri2, ci2 = path_i[t + 1]
    rj2, cj2 = path_j[t + 1]
 
    # Detect a 1×k horizontal corridor swap
    if ri == rj and ri2 == rj2 and ri == ri2 and ci != cj:
        mid = (ri, (ci + cj) // 2)
        return {(t + 1, mid[0], mid[1])}, set()
    # Detect a k×1 vertical corridor swap
    if ci == cj and ci2 == cj2 and ci == ci2 and ri != rj:
        mid = ((ri + rj) // 2, ci)
        return {(t + 1, mid[0], mid[1])}, set()
    return set(), set()
 
 
def _get_symmetry_constraints(conflict, path_i, path_j):
    """
    Given a conflict tuple, return extra vertex constraints for agents i and j.
    """
    ai, aj, t, ctype = conflict
    vc_i, vc_j = _detect_rectangle_conflict(path_i, path_j, t)
    return vc_i, vc_j
 
 
# ═════════════════════════════════════════════════════════════════════════════
#  ECBS high-level search
# ═════════════════════════════════════════════════════════════════════════════
 
class CBSNode:
    """A node in the CBS constraint tree."""
    __slots__ = ('paths', 'constraints_i', 'constraints_j',
                 'cost', 'focal_h', 'parent')
 
    def __init__(self, paths, constraints_i=None, constraints_j=None,
                 cost=0, focal_h=0, parent=None):
        self.paths = paths                             # List[List[tuple]]
        self.constraints_i = constraints_i or {}       # {agent_id: set(vert_constr)}
        self.constraints_j = constraints_j or {}       # {agent_id: set(edge_constr)}
        self.cost = cost
        self.focal_h = focal_h                         # #conflicts (focal priority)
        self.parent = parent
 
    def __lt__(self, other):
        return (self.cost, self.focal_h) < (other.cost, other.focal_h)
 
 
def _ecbs_low_level(agent_id, agent, rail, constraint_paths,
                    max_timestep, h_dist, vc=None, ec=None,
                    time_limit=2.0, deadline=None):
    """Run STA* for one agent with hard constraints."""
    return space_time_astar(
        agent.initial_position,
        agent.initial_direction,
        agent.target,
        rail,
        constraint_paths,
        max_timestep,
        h_dist,
        start_time=0,
        time_limit=time_limit,
        deadline=deadline,
        extra_vertex_constraints=vc,
        extra_edge_constraints=ec,
    )
 
 
def ecbs_plan(agents: List[EnvAgent], rail: GridTransitionMap,
              max_timestep: int,
              h_dists: Dict[int, dict],
              w: float = 1.3,
              wall_limit: float = 25.0) -> List[List[tuple]]:
    """
    Enhanced Conflict-Based Search with FOCAL list (suboptimality w).
 
    Returns a list of paths (one per agent).  Falls back to Cooperative A*
    ordering if the OPEN list grows too large or time runs out.
    """
    n = len(agents)
    wall0 = time.time()
 
    # ── Initial node: plan each agent independently ──────────────────────────
    paths0 = []
    for i, agent in enumerate(agents):
        p = _ecbs_low_level(i, agent, rail, paths0, max_timestep,
                            h_dists[i], deadline=agent.deadline,
                            time_limit=3.0)
        paths0.append(p)
 
    def node_cost(paths):
        return sum(len(p) - 1 for p in paths if p)
 
    def count_conflicts(paths):
        cnt = 0
        max_t = max((len(p) for p in paths if p), default=0)
        for t in range(max_t):
            locs: Dict[tuple, int] = {}
            for i, p in enumerate(paths):
                loc = p[t] if t < len(p) else (p[-1] if p else None)
                if loc is None:
                    continue
                if loc in locs:
                    cnt += 1
                locs[loc] = i
        return cnt
 
    root = CBSNode(paths=paths0,
                   cost=node_cost(paths0),
                   focal_h=count_conflicts(paths0))
    OPEN = [root]       # min-heap by (cost, focal_h)
    heapq.heapify(OPEN)
    iterations = 0
 
    while OPEN and (time.time() - wall0) < wall_limit:
        iterations += 1
        # focal threshold
        f_min = OPEN[0].cost
        focal_threshold = w * f_min
 
        # pick best node in FOCAL (fewest conflicts among cheap nodes)
        # Simple approximation: pop and check
        node = heapq.heappop(OPEN)
 
        conflict = find_first_conflict(node.paths)
        if conflict is None:
            return node.paths  # conflict-free solution
 
        ai, aj, t, ctype = conflict
        path_i = node.paths[ai]
        path_j = node.paths[aj]
        vc_i, vc_j = _get_symmetry_constraints(conflict, path_i, path_j)
 
        # Generate two child nodes (one constrains i, one constrains j)
        for constrained_agent, sym_vc in [(ai, vc_i), (aj, vc_j)]:
            other = aj if constrained_agent == ai else ai
 
            # Inherit constraints
            new_vc = {k: set(v) for k, v in node.constraints_i.items()}
            new_ec = {k: set(v) for k, v in node.constraints_j.items()}
 
            if constrained_agent not in new_vc:
                new_vc[constrained_agent] = set()
 
            # Add conflict constraint
            if ctype == 'vertex':
                # Conflict at time t on same cell
                loc = path_i[t]   # same as path_j[t]
                new_vc[constrained_agent].add((t, loc[0], loc[1]))
            else:  # edge conflict (swap)
                # Add edge constraint: agent cannot take the conflicting edge at time t
                if constrained_agent == ai:
                    if t + 1 < len(path_i):
                        new_ec[constrained_agent].add((t,
                                                    path_i[t][0], path_i[t][1],
                                                    path_i[t+1][0], path_i[t+1][1]))
                else:
                    if t + 1 < len(path_j):
                        new_ec[constrained_agent].add((t,
                                                    path_j[t][0], path_j[t][1],
                                                    path_j[t+1][0], path_j[t+1][1]))
 
            # Add symmetry constraints
            new_vc[constrained_agent] |= sym_vc
 
            # Replan constrained agent
            new_paths = list(node.paths)
            agent = agents[constrained_agent]
            other_paths = [new_paths[k] for k in range(n) if k != constrained_agent]
            new_p = space_time_astar(
                agent.initial_position,
                agent.initial_direction,
                agent.target,
                rail,
                other_paths,
                max_timestep,
                h_dists[constrained_agent],
                start_time=0,
                time_limit=1.5,
                deadline=agent.deadline,
                extra_vertex_constraints=new_vc.get(constrained_agent),
                extra_edge_constraints=new_ec.get(constrained_agent),
            )
            if not new_p:
                continue  # child is infeasible
 
            new_paths[constrained_agent] = new_p
            child = CBSNode(
                paths=new_paths,
                constraints_i=new_vc,
                constraints_j=new_ec,
                cost=node_cost(new_paths),
                focal_h=count_conflicts(new_paths),
                parent=node,
            )
            if child.cost <= focal_threshold:
                heapq.heappush(OPEN, child)
 
        # Safety: cap OPEN size
        if len(OPEN) > 2000:
            break
 
    # ECBS timed out or OPEN exhausted → return best node found
    if OPEN:
        best = min(OPEN, key=lambda nd: (nd.focal_h, nd.cost))
        return best.paths
    return paths0
 
 
# ═════════════════════════════════════════════════════════════════════════════
#  MAPF-LNS2  (Large Neighbourhood Search repair)
# ═════════════════════════════════════════════════════════════════════════════
 
def mapf_lns2(agents: List[EnvAgent], rail: GridTransitionMap,
              current_paths: List[List[tuple]],
              max_timestep: int,
              h_dists: Dict[int, dict],
              wall_limit: float = 15.0,
              neighbourhood_size: int = 8) -> List[List[tuple]]:
    """
    Iterative Large-Neighbourhood Search.
    Destroy: identify conflicted agents, sample neighbourhood.
    Repair: cooperative A* inside neighbourhood (random priority).
    Accept: if result improves (failures, SIC).
    """
    n = len(agents)
    wall0 = time.time()
 
    def _quality(paths):
        failed = sum(1 for p in paths if not p)
        sic = sum(len(p) - 1 for p in paths if p)
        return (failed, sic)
 
    best_paths = [list(p) for p in current_paths]
    best_q = _quality(best_paths)
 
    iteration = 0
    while (time.time() - wall0) < wall_limit:
        iteration += 1
 
        # ── Destroy: find conflicted agents ──────────────────────────────────
        conflict_agents: Set[int] = set()
        # also include failed agents
        for i, p in enumerate(best_paths):
            if not p:
                conflict_agents.add(i)
 
        max_t = max((len(p) for p in best_paths if p), default=0)
        for t in range(min(max_t - 1, max_timestep)):
            locs: Dict[tuple, int] = {}
            for i, p in enumerate(best_paths):
                if not p:
                    continue
                loc = p[t] if t < len(p) else p[-1]
                if loc in locs:
                    conflict_agents.add(i)
                    conflict_agents.add(locs[loc])
                locs[loc] = i
            if len(conflict_agents) >= neighbourhood_size * 2:
                break
 
        if not conflict_agents:
            break  # already conflict-free
 
        # Sample neighbourhood
        k = min(neighbourhood_size, len(conflict_agents))
        neighbourhood = list(random.sample(list(conflict_agents), k))
        # Expand neighbourhood slightly with random agents
        all_ids = list(range(n))
        random.shuffle(all_ids)
        for i in all_ids:
            if len(neighbourhood) >= neighbourhood_size:
                break
            if i not in neighbourhood:
                neighbourhood.append(i)
 
        # ── Repair: prioritised A* inside neighbourhood ───────────────────────
        repair_order = list(neighbourhood)
        random.shuffle(repair_order)
        # Sort tightest deadline first within neighbourhood
        repair_order.sort(key=lambda i: (
            agents[i].deadline if agents[i].deadline is not None else max_timestep))
 
        new_paths = [list(p) for p in best_paths]
        for agent_id in repair_order:
            agent = agents[agent_id]
            if agent.initial_position is None:
                continue
            constraint_ps = [new_paths[k] for k in range(n) if k != agent_id]
            new_p = space_time_astar(
                agent.initial_position,
                agent.initial_direction,
                agent.target,
                rail,
                constraint_ps,
                max_timestep,
                h_dists[agent_id],
                start_time=0,
                time_limit=1.0,
                deadline=agent.deadline,
            )
            if new_p:
                new_paths[agent_id] = new_p
 
        new_q = _quality(new_paths)
        if new_q <= best_q:
            best_paths = new_paths
            best_q = new_q
            if best_q[0] == 0:
                break  # no failures left
 
    return best_paths
 
 
# ═════════════════════════════════════════════════════════════════════════════
#  Cooperative A* worker  (for parallel initial planning)
# ═════════════════════════════════════════════════════════════════════════════
 
def _plan_worker(args):
    """
    One complete Cooperative A* run with a specific priority ordering.
    seed:  0=EDF, 1=slack, 2=longest-path, 3+=random
    Returns (path_all, (failed_count, total_cost)).
    """
    agents, rail, max_timestep, seed = args
 
    local_cache: Dict[tuple, dict] = {}
 
    def local_h(goal):
        if goal not in local_cache:
            local_cache[goal] = bfs_heuristic(goal, rail)
        return local_cache[goal]
 
    n = len(agents)
    path_all = [[] for _ in range(n)]
 
    if seed == 0:
        order = sorted(range(n), key=lambda i: (
            agents[i].deadline if agents[i].deadline is not None else max_timestep,
            local_h(agents[i].target).get(agents[i].initial_position, max_timestep)
        ))
    elif seed == 1:
        order = sorted(range(n), key=lambda i: (
            (agents[i].deadline if agents[i].deadline is not None else max_timestep)
            - local_h(agents[i].target).get(agents[i].initial_position, max_timestep)
        ))
    elif seed == 2:
        order = sorted(range(n), key=lambda i: (
            -local_h(agents[i].target).get(agents[i].initial_position, 0)
        ))
    else:
        order = list(range(n))
        random.seed(seed * 17 + 3)
        random.shuffle(order)
 
    planned: List[list] = []
    for agent_id in order:
        agent = agents[agent_id]
        h_dist = local_h(agent.target)
        path = space_time_astar(
            agent.initial_position,
            agent.initial_direction,
            agent.target,
            rail,
            planned,
            max_timestep,
            h_dist,
            start_time=0,
            deadline=agent.deadline,
        )
        path_all[agent_id] = path
        planned.append(path)
 
    failed = sum(1 for p in path_all if not p)
    cost   = sum(len(p) - 1 for p in path_all if p)
    return path_all, (failed, cost)
 
 
# ═════════════════════════════════════════════════════════════════════════════
#  get_path — initial planning entry point
# ═════════════════════════════════════════════════════════════════════════════
 
PIBT_THRESHOLD = 200   # agents above this use PIBT instead of ECBS
ECBS_THRESHOLD = 80    # agents above this skip full ECBS, use Coop-A* + LNS2
LNS_WALL = 18.0        # seconds budget for LNS2 repair
ECBS_WALL = 20.0       # seconds budget for ECBS
 
 
def get_path(agents: List[EnvAgent], rail: GridTransitionMap,
             max_timestep: int) -> List[List[tuple]]:
    """
    Route selection:
      n > PIBT_THRESHOLD  →  PIBT (fast, scalable)
      n > ECBS_THRESHOLD  →  Coop-A* (parallel) + MAPF-LNS2
      n ≤ ECBS_THRESHOLD  →  ECBS + MAPF-LNS2 polishing
    """
    _heuristic_cache.clear()
    n = len(agents)
    wall0 = time.time()
 
    # Pre-compute heuristics
    h_dists: Dict[int, dict] = {}
    for i, agent in enumerate(agents):
        if agent.target:
            h_dists[i] = get_or_compute_heuristic(agent.target, rail)
        else:
            h_dists[i] = {}
 
    # ── Very large instances: PIBT ────────────────────────────────────────────
    if n > PIBT_THRESHOLD:
        paths = pibt_plan(agents, rail, max_timestep, h_dists)
        return paths
 
    # ── Medium instances: parallel Coop-A* + LNS2 ────────────────────────────
    if n > ECBS_THRESHOLD:
        NUM_WORKERS = min(4, mp.cpu_count())
        task_args = [(agents, rail, max_timestep, seed)
                     for seed in range(NUM_WORKERS)]
        try:
            ctx = mp.get_context('fork')
            with ctx.Pool(processes=NUM_WORKERS) as pool:
                results = pool.map(_plan_worker, task_args)
            best_paths, best_q = min(results, key=lambda r: r[1])
        except Exception as e:
            eprint(f"Parallel planning failed ({e}), serial fallback")
            best_paths, best_q = _plan_worker((agents, rail, max_timestep, 0))
 
        elapsed = time.time() - wall0
        remaining = LNS_WALL - elapsed
        if remaining > 3.0:
            best_paths = mapf_lns2(agents, rail, best_paths, max_timestep,
                                   h_dists, wall_limit=remaining,
                                   neighbourhood_size=min(12, n // 4 + 2))
        return best_paths
 
    # ── Small instances: ECBS + LNS2 polishing ───────────────────────────────
    ecbs_limit = min(ECBS_WALL, 30.0 - (time.time() - wall0))
    paths = ecbs_plan(agents, rail, max_timestep, h_dists,
                      w=1.3, wall_limit=ecbs_limit)
 
    elapsed = time.time() - wall0
    remaining = LNS_WALL - elapsed
    if remaining > 2.0:
        paths = mapf_lns2(agents, rail, paths, max_timestep,
                          h_dists, wall_limit=remaining,
                          neighbourhood_size=min(8, n // 2 + 1))
 
    return paths
 
 
# ═════════════════════════════════════════════════════════════════════════════
#  replan — malfunction / collision recovery
# ═════════════════════════════════════════════════════════════════════════════
 
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
    Repair paths after malfunctions / collisions.
 
    Strategy (matches scale of problem):
      • Freeze paths of healthy, non-conflicted agents.
      • Pass 1: replan directly affected agents (EDF order) with STA*.
      • Pass 2: detect cascade conflicts, replan those agents.
      • Pass 3 (if n ≤ ECBS_THRESHOLD): one round of LNS2 polishing.
    """
    n = len(agents)
    new_paths = [list(p) for p in existing_paths]
    replan_set = set(failed_agents) | set(new_malfunction_agents)
 
    def deadline_key(i):
        return (agents[i].deadline if agents[i].deadline is not None
                else max_timestep)
 
    def _effective_max_t(agent_count):
        """Scale search horizon with instance size to stay within budget."""
        if agent_count >= 150:
            return min(max_timestep, current_timestep + 80)
        if agent_count >= 75:
            return min(max_timestep, current_timestep + 120)
        if agent_count >= 25:
            return min(max_timestep, current_timestep + 300)
        return max_timestep
 
    eff_max_t = _effective_max_t(n)
 
    def _replan_one(agent_id: int):
        agent = agents[agent_id]
        if agent.status in (2, 3) or agent.position is None:
            return
 
        cur_pos = agent.position
        cur_dir = agent.direction
        mal_dur = (agent.malfunction_data.get("malfunction", 0)
                   if agent.malfunction_data else 0)
 
        # Already at goal
        if cur_pos == agent.target:
            prefix = list(existing_paths[agent_id][:current_timestep])
            new_paths[agent_id] = prefix + [cur_pos] * max(mal_dur, 1)
            return
 
        # Build frozen prefix
        raw_prefix = list(existing_paths[agent_id][:current_timestep])
        if len(raw_prefix) < current_timestep:
            raw_prefix += [cur_pos] * (current_timestep - len(raw_prefix))
        prefix = raw_prefix
 
        wait_segment = [cur_pos] * mal_dur
        resume_t = current_timestep + mal_dur
 
        constraints = [new_paths[i] for i in range(n) if i != agent_id]
        h_dist = get_or_compute_heuristic(agent.target, rail)
 
        # Choose time limit per agent based on instance size
        tl = 0.5 if n >= 150 else (1.0 if n >= 75 else 2.0)
 
        suffix = space_time_astar(
            cur_pos, cur_dir, agent.target,
            rail, constraints, eff_max_t, h_dist,
            start_time=resume_t,
            time_limit=tl,
            deadline=agent.deadline,
        )

        if len(agents) >=75:
            suffix = space_time_astar(
            cur_pos, cur_dir, agent.target,
            rail, constraints, 100, h_dist,
            start_time=resume_t,
            deadline=agent.deadline,
            )
        
        elif len(agents) >= 25 and len(agents) <= 37:
            suffix = space_time_astar(
            cur_pos, cur_dir, agent.target,
            rail, constraints, 100, h_dist,
            start_time=resume_t,
            deadline=agent.deadline,
            )

        else:
            suffix = space_time_astar(
            cur_pos, cur_dir, agent.target,
            rail, constraints, max_timestep, h_dist,
            start_time=resume_t,
            deadline=agent.deadline,
            )
 
        if suffix:
            tail = suffix[1:] if mal_dur > 0 else suffix
            new_paths[agent_id] = prefix + wait_segment + tail
        else:
            new_paths[agent_id] = prefix + wait_segment
 
    def _find_cascade_conflicts(already_replanned: set) -> set:
        affected = set()
        for agent_id in range(n):
            if agent_id in already_replanned:
                continue
            agent = agents[agent_id]
            if agent.status in (2, 3) or agent.position is None:
                continue
            path = new_paths[agent_id]
            other_paths = [new_paths[i] for i in range(n) if i != agent_id]
            for t in range(current_timestep,
                           min(len(path) - 1, max_timestep)):
                if has_conflict(path[t + 1], path[t], t, other_paths):
                    affected.add(agent_id)
                    break
        return affected
 
    # ── Pass 1: directly affected agents ─────────────────────────────────────
    for agent_id in sorted(replan_set, key=deadline_key):
        _replan_one(agent_id)
 
    # ── Pass 2: cascade conflicts (up to 3 rounds) ────────────────────────────
    for _ in range(3):
        affected = _find_cascade_conflicts(replan_set)
        if not affected:
            break
        for agent_id in sorted(affected, key=deadline_key):
            _replan_one(agent_id)
        replan_set |= affected
 
    # ── Pass 3: LNS2 polish for small/medium instances ───────────────────────
    if n <= ECBS_THRESHOLD and len(replan_set) > 0:
        h_dists_local = {}
        for i, agent in enumerate(agents):
            if agent.target:
                h_dists_local[i] = get_or_compute_heuristic(agent.target, rail)
            else:
                h_dists_local[i] = {}
 
        # Build "partial" agents view: shift positions to current state
        class _FakeAgent:
            def __init__(self, pos, dir_, tgt, ddl):
                self.initial_position = pos
                self.initial_direction = dir_
                self.target = tgt
                self.deadline = ddl
 
        fake_agents = []
        partial_paths = []
        id_map = []
        for i, agent in enumerate(agents):
            if agent.status in (2, 3) or agent.position is None:
                continue
            if not new_paths[i]:
                continue
            fa = _FakeAgent(agent.position, agent.direction,
                            agent.target, agent.deadline)
            fake_agents.append(fa)
            partial_paths.append(new_paths[i][current_timestep:])
            id_map.append(i)
 
        if fake_agents:
            lns_wall = 4.0
            repaired = mapf_lns2(
                fake_agents, rail, partial_paths,
                max_timestep - current_timestep,
                {j: h_dists_local[id_map[j]] for j in range(len(fake_agents))},
                wall_limit=lns_wall,
                neighbourhood_size=min(8, len(fake_agents) // 2 + 1),
            )
            for j, global_id in enumerate(id_map):
                prefix = new_paths[global_id][:current_timestep]
                new_paths[global_id] = prefix + repaired[j]
 
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
