"""
train_priority_nn.py
====================
Trains AgentPriorityNet using outcome-based labels derived from running
cooperative A* on each instance.
Label logic (1 ordering per instance):
  - Run coop A* with the best analytic ordering (least-slack-first = seed 1).
  - For each agent, compute:
        conflict_delay = actual_path_cost - bfs_dist_to_goal
  - Agents with HIGH conflict_delay were hurt by lower-priority agents blocking
    them. They should have gone EARLIER. So priority rank = argsort(delay)
    descending — most-delayed agents get rank 0 (highest priority).
  - This teaches the NN: "in instances like this, agents with these features
    tend to get blocked — push them to the front."
Why 1 ordering instead of many:
  - With 56 instances and a per-agent time limit you can afford ~5-10s per
    instance for A*, giving clean signal without hours of offline training.
  - A second ordering (random) is optionally run as a contrastive check to
    confirm the signal is real, but labels always come from the best plan.
"""
import argparse
import glob
import os
import pickle
import sys
import time
import heapq
import json
import random
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
from flatland.core.grid.rail_env_grid import RailEnvTransitions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nn_priority import AgentPriorityNet, extract_features, FEATURE_DIM
try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.envs.agent_utils import EnvAgent
    from flatland.utils.controller import get_action, Train_Actions, Directions
except ImportError as e:
    print(f"ERROR: Cannot import flatland: {e}")
    sys.exit(1)
# ═══════════════════════════════════════════════════════════════════════════
#  Rail helpers (unchanged)
# ═══════════════════════════════════════════════════════════════════════════
def get_rail_transitions(rail, x: int, y: int, direction: int):
    if hasattr(rail, 'get_transitions'):
        return rail.get_transitions(x, y, direction)
    cell_code = rail[x][y]
    transitions = RailEnvTransitions()
    return transitions.get_transitions(cell_code, direction)
def bfs_heuristic(goal: tuple, rail) -> dict:
    dist = {goal: 0}
    q = deque([goal])
    while q:
        x, y = q.popleft()
        for d in range(4):
            for action, valid in enumerate(get_rail_transitions(rail, x, y, d)):
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
def has_conflict(new_loc, cur_loc, t, constraint_paths):
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
def space_time_astar(start, start_dir, goal, rail, constraint_paths,
                     max_timestep, h_dist, start_time=0,
                     deadline=None, time_limit=5.0) -> list:
    if start not in h_dist:
        return []
    deadline_wall = time.time() + time_limit
    open_heap = [(h_dist[start], 0, start[0], start[1], start_dir, start_time)]
    visited = set()
    best_g = {(start[0], start[1], start_dir, start_time): 0}
    parent = {(start[0], start[1], start_dir, start_time): None}
    exp = 0
    while open_heap:
        exp += 1
        if exp % 200 == 0 and time.time() > deadline_wall:
            # Timeout — return a wait-in-place path as graceful fallback
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
            if ns not in best_g or new_g < best_g[ns]:
                best_g[ns] = new_g
                parent[ns] = state
                new_h = h_dist.get((nx, ny))
                if new_h is None:
                    return
                delay = max(0, new_t - deadline) if deadline is not None else 0
                heapq.heappush(open_heap,
                               (new_g + new_h + 2 * delay, new_g, nx, ny, new_dir, new_t))
        for action, valid in enumerate(get_rail_transitions(rail, x, y, direction)):
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
def coop_astar(agents, rail, max_timestep, order, h_dists, deadlines,
               per_agent_limit: float = 5.0) -> List[list]:
    """Run cooperative A* in the given agent order. Returns path list indexed by agent id."""
    n = len(agents)
    path_all = [[] for _ in range(n)]
    planned: List[list] = []
    for agent_id in order:
        ag = agents[agent_id]
        if ag.initial_position is None or ag.target is None:
            planned.append([])
            continue
        path = space_time_astar(
            ag.initial_position, ag.initial_direction, ag.target,
            rail, planned, max_timestep, h_dists[agent_id],
            deadline=deadlines[agent_id],
            time_limit=per_agent_limit,
        )
        path_all[agent_id] = path
        planned.append(path)
    return path_all
def evaluate_paths(paths, agents, max_timestep, deadlines) -> Tuple[int, int]:
    failed, cost = 0, 0
    for i, p in enumerate(paths):
        if not p:
            failed += 1
            cost += max_timestep
            continue
        path_cost = len(p) - 1
        ddl = deadlines[i] if i < len(deadlines) else max_timestep
        cost += path_cost + 2 * max(0, path_cost - ddl)
    return failed, cost
# ═══════════════════════════════════════════════════════════════════════════
#  Instance loading (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════
def load_instance(pkl_path: str, ddl_path: Optional[str] = None):
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"  [WARN] Cannot load {pkl_path}: {e}")
        return None
    try:
        if isinstance(data, dict):
            agents = data.get('agents') or data.get('agent_list') or []
            rail   = data.get('rail') or data.get('grid') or data.get('transition_map')
            tmax   = data.get('max_timestep') or data.get('T_max') or 300
        elif isinstance(data, (tuple, list)) and len(data) >= 2:
            first = data[0]
            if hasattr(first, 'agents'):
                env    = first
                agents = env.agents
                rail   = env.rail
                tmax   = getattr(env, '_max_episode_steps', 300)
            else:
                agents, rail = data[0], data[1]
                tmax = data[2] if len(data) > 2 else 300
        elif hasattr(data, 'agents'):
            agents = data.agents
            rail   = data.rail
            tmax   = getattr(data, '_max_episode_steps', 300)
        else:
            print(f"  [WARN] Unrecognised pkl format in {pkl_path}")
            return None
        if agents is None or rail is None:
            return None
    except Exception as e:
        print(f"  [WARN] Parsing failed for {pkl_path}: {e}")
        return None
    deadlines = [int(tmax)] * len(agents)
    if ddl_path and os.path.exists(ddl_path):
        try:
            with open(ddl_path, 'r') as f:
                ddl_data = json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            try:
                with open(ddl_path, 'rb') as f:
                    ddl_data = pickle.load(f)
            except Exception:
                try:
                    ddl_data = np.load(ddl_path)
                    if isinstance(ddl_data, np.ndarray):
                        ddl_data = ddl_data.tolist()
                except Exception as e:
                    print(f"  Could not load deadlines: {e}")
                    ddl_data = None
        if ddl_data is not None:
            for i in range(min(len(agents), len(ddl_data))):
                deadlines[i] = ddl_data[i]
    return agents, rail, int(tmax), deadlines
# ═══════════════════════════════════════════════════════════════════════════
#  Outcome-based label generation  ← the key rewrite
# ═══════════════════════════════════════════════════════════════════════════
def outcome_labels(agents, paths, h_dists, max_timestep,
                   deadlines: List[int]) -> np.ndarray:
    """
    Convert actual A* paths into priority ranks.

    For each agent:
        conflict_delay = actual_path_cost  -  bfs_dist_to_goal

    Agents with large conflict_delay were blocked by earlier agents and should
    have been planned first. We rank by descending delay so rank=0 means
    "should go first".

    Ties (delay==0, all agents reached without conflict) fall back to
    ascending slack so the labels remain informative.

    deadlines: per-agent deadline list loaded from .ddl files — must be the
    same list passed to extract_features() so that training features and
    labels are consistent (and consistent with how question3.py uses
    agent.deadline at inference time).
    """
    n = len(agents)
    delays = np.zeros(n)
    slacks = np.zeros(n)
    for i, ag in enumerate(agents):
        pos = ag.initial_position
        if pos is None:
            continue
        bfs_d = h_dists[i].get(pos, max_timestep)
        p = paths[i]
        actual_cost = (len(p) - 1) if p else max_timestep
        delays[i] = actual_cost - bfs_d          # 0 if no conflict at all
        # FIX: use the deadlines list (same source as extract_features) rather
        # than ag.deadline which may be absent on some Flatland agent types.
        ddl = deadlines[i] if i < len(deadlines) else max_timestep
        slacks[i] = ddl - bfs_d
    # Primary sort: descending delay (most blocked → lowest rank index → goes first)
    # Secondary sort: ascending slack (tightest deadline → goes first when no conflict)
    keys = list(zip(-delays, slacks, range(n)))
    keys.sort()
    ranks = np.zeros(n)
    for rank, (_, _, agent_id) in enumerate(keys):
        ranks[agent_id] = rank / max(1, n - 1)
    return ranks
def generate_training_data(pkl_files: List[str], ddl_files: List[str],
                            per_agent_limit: float = 5.0):
    """
    For each instance:
      1. Run coop A* with least-slack-first ordering (the best analytic heuristic).
      2. Measure actual conflict delays per agent.
      3. Emit (features, outcome_rank) as one training sample per agent.
    per_agent_limit: seconds of A* search budget per agent during data generation.
    """
    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    # Create cache directory
    cache_dir = "training_cache/"
    os.makedirs(cache_dir, exist_ok=True)

    for pkl_path, ddl_path in zip(pkl_files, ddl_files):
        name = os.path.basename(pkl_path)
        # Create cache filename based on input file
        cache_name = os.path.basename(pkl_path).replace('.pkl', '_features.npz')
        cache_path = os.path.join(cache_dir, cache_name)
        
        # Check if cached version exists
        if os.path.exists(cache_path):
            print(f"  Loaded from cache: {name}")
            cached = np.load(cache_path)
            all_X.append(cached['X'])
            all_y.append(cached['y'])
            continue  # Skip A* computation for this file
        
        print(f"  {name} …", end="", flush=True)
        t0 = time.time()
        result = load_instance(pkl_path, ddl_path)
        if result is None:
            print(" SKIP"); continue
        agents, rail, max_timestep, deadlines = result
        n = len(agents)

        if n <= 1:
            print(" SKIP (single agent — no conflict signal)"); continue
        # ── BFS distances (reused for features + labels) ──────────────────
        goal_cache: dict = {}
        h_dists: Dict[int, dict] = {}
        for i, ag in enumerate(agents):
            if ag.target is not None:
                if ag.target not in goal_cache:
                    goal_cache[ag.target] = bfs_heuristic(ag.target, rail)
                h_dists[i] = goal_cache[ag.target]
            else:
                h_dists[i] = {}
        if hasattr(rail, 'height'):
            grid_rows, grid_cols = rail.height, rail.width
        elif hasattr(rail, '__len__'):
            grid_rows = len(rail)
            grid_cols = len(rail[0]) if grid_rows > 0 else 50
        else:
            grid_rows, grid_cols = 50, 50
        # ── Analytic ordering: least-slack-first (= seed 1 in question3.py) ──
        def bfs_dist(i):
            pos = agents[i].initial_position
            return h_dists[i].get(pos, max_timestep) if pos is not None else max_timestep
        order = sorted(range(n), key=lambda i: (deadlines[i] - bfs_dist(i),
                                                 deadlines[i],
                                                 -bfs_dist(i)))
        # ── Run A* and collect actual paths ───────────────────────────────
        paths = coop_astar(agents, rail, max_timestep, order,
                           h_dists, deadlines, per_agent_limit=per_agent_limit)
        failed, cost = evaluate_paths(paths, agents, max_timestep, deadlines)
        # ── Build labels from actual outcomes ─────────────────────────────
        # FIX: pass the loaded deadlines list so outcome_labels and
        # extract_features use the exact same deadline values — this is also
        # consistent with question3.py which reads ag.deadline at inference
        # time (set from the same .ddl source via the evaluator harness).
        ranks = outcome_labels(agents, paths, h_dists, max_timestep, deadlines)
        # ── Features ──────────────────────────────────────────────────────
        feats = extract_features(agents, h_dists, max_timestep,
                                 grid_rows, grid_cols, deadlines)
        all_X.append(feats)
        all_y.append(ranks)
        elapsed = time.time() - t0
        print(f" OK  (n={n}, failed={failed}, cost={cost}, {elapsed:.1f}s)")
    if not all_X:
        print("[WARN] No training data generated!")
        return np.zeros((0, FEATURE_DIM)), np.zeros(0)
    return np.vstack(all_X), np.concatenate(all_y)
# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Train AgentPriorityNet using outcome-based labels from actual A* runs."
    )
    parser.add_argument("--test_dir",        default="multi_test_case/")
    parser.add_argument("--single_dir",      default="single_test_case/")
    parser.add_argument("--ddl_dir",         default=None)
    parser.add_argument("--output",          default="model_weights.npz")
    parser.add_argument("--epochs",          type=int,   default=200)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--batch_size",      type=int,   default=64)
    parser.add_argument("--per_agent_limit", type=float, default=5.0,
                        help="A* time budget (seconds) per agent during data generation")
    args = parser.parse_args()
    # ── Collect pkl files ─────────────────────────────────────────────────
    pkl_files: List[str] = []
    ddl_files: List[str] = []
    for search_dir in [args.test_dir, args.single_dir]:
        if search_dir and os.path.isdir(search_dir):
            found = sorted(glob.glob(os.path.join(search_dir, "level*_test_*.pkl")))
            ddl_dir = args.ddl_dir or search_dir
            for f in found:
                pkl_files.append(f)
                ddl_candidate = os.path.join(
                    ddl_dir, os.path.basename(f).replace(".pkl", ".ddl"))
                ddl_files.append(ddl_candidate if os.path.exists(ddl_candidate) else "")
    # FIX: deduplicate by resolved absolute path so that overlapping or
    # identical --test_dir / --single_dir arguments don't cause the same
    # instance to be trained on twice, which would silently bias the model.
    seen: set = set()
    deduped_pkl: List[str] = []
    deduped_ddl: List[str] = []
    for p, d in zip(pkl_files, ddl_files):
        key = os.path.realpath(p)
        if key not in seen:
            seen.add(key)
            deduped_pkl.append(p)
            deduped_ddl.append(d)
    pkl_files, ddl_files = deduped_pkl, deduped_ddl
    if not pkl_files:
        print("ERROR: No .pkl files found.")
        sys.exit(1)
    print(f"Found {len(pkl_files)} instances.")
    # ── Generate training data ────────────────────────────────────────────
    print("\n=== Generating outcome-based training data ===")
    X, y = generate_training_data(pkl_files, ddl_files,
                                  per_agent_limit=args.per_agent_limit)
    print(f"\nTotal training samples: {X.shape[0]}")
    if X.shape[0] == 0:
        print("No training data — saving default weights.")
        AgentPriorityNet().save(args.output)
        return
    # ── Train ─────────────────────────────────────────────────────────────
    print("\n=== Training ===")
    net = AgentPriorityNet()
    n_total = X.shape[0]
    bs = min(args.batch_size, n_total)
    rng = np.random.default_rng(0)
    for epoch in range(1, args.epochs + 1):
        idx = rng.permutation(n_total)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_total, bs):
            batch_idx = idx[start:start + bs]
            Xb, yb = X[batch_idx], y[batch_idx]
            # Pairwise ranking loss throughout — the labels are ordinal ranks,
            # not regression targets, so pairwise loss is always the right choice.
            loss = net.pairwise_train_step(Xb, yb, lr=args.lr)
            epoch_loss += loss
            n_batches += 1
        if epoch % 20 == 0 or epoch == 1:
            avg_loss = epoch_loss / max(1, n_batches)
            # Pairwise accuracy on full dataset
            scores = net.forward(X)
            correct = total = 0
            sample_pairs = min(500, n_total * (n_total - 1) // 2)
            for _ in range(sample_pairs):
                i, j = rng.choice(n_total, 2, replace=False)
                if abs(y[i] - y[j]) > 0.05:
                    correct += int((scores[i] < scores[j]) == (y[i] < y[j]))
                    total += 1
            # FIX: guard against 0/0 when all sampled rank differences fall
            # below the 0.05 threshold (can happen on small or homogeneous
            # datasets).  Report "n/a" rather than a misleading 0.000.
            acc_str = f"{correct / total:.3f}" if total > 0 else "n/a"
            print(f"  Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  "
                  f"pair-acc={acc_str}")
    net.save(args.output)
    print(f"\nDone. Saved → {args.output}")
if __name__ == "__main__":
    main()