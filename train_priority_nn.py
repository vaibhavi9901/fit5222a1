import argparse
import glob
import os
import pickle
import sys
import time
import heapq
import random
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
from flatland.core.grid.rail_env_grid import RailEnvTransitions

# ── import our model ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nn_priority import AgentPriorityNet, extract_features, FEATURE_DIM

# ── flatland imports ──────────────────────────────────────────────────────────
try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.envs.agent_utils import EnvAgent
    from flatland.utils.controller import get_action, Train_Actions, Directions
except ImportError as e:
    print(f"ERROR: Cannot import flatland: {e}")
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
#  Minimal Cooperative A* (self-contained for training — no side effects)
# ════════════════════════════════════════════════════════════════════════════

def get_rail_transitions(rail, x: int, y: int, direction: int):
    """Return a 4‑tuple of booleans indicating allowed moves from (x,y) facing `direction`."""
    # Case 1: rail is a proper GridTransitionMap
    if hasattr(rail, 'get_transitions'):
        return rail.get_transitions(x, y, direction)
    
    # Case 2: rail is a 2D list/array of transition codes (integers)
    # Use RailEnvTransitions to decode the cell
    cell_code = rail[x][y]          # note: rail is stored as [row][col] (x = row, y = col)
    transitions = RailEnvTransitions()
    return transitions.get_transitions(cell_code, direction)

def bfs_heuristic(goal: tuple, rail) -> dict:
    dist = {goal: 0}
    q = deque([goal])
    while q:
        x, y = q.popleft()
        for d in range(4):
            transitions = get_rail_transitions(rail, x, y, d)
            for action, valid in enumerate(transitions):
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
    t0 = time.time()
    open_heap = [(h_dist[start], 0, start[0], start[1], start_dir, start_time)]
    visited = set()
    best_g: Dict[tuple, int] = {(start[0], start[1], start_dir, start_time): 0}
    parent: Dict[tuple, Optional[tuple]] = {
        (start[0], start[1], start_dir, start_time): None
    }
    exp = 0
    while open_heap:
        exp += 1
        if exp % 500 == 0 and time.time() - t0 > time_limit:
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


def coop_astar(agents, rail, max_timestep, order, h_dists, deadlines) -> List[list]:
    n = len(agents)
    path_all = [[] for _ in range(n)]
    planned: List[list] = []
    for agent_id in order:
        ag = agents[agent_id]
        if ag.initial_position is None or ag.target is None:
            planned.append([])
            continue
        hd = h_dists[agent_id]
        path = space_time_astar(
            ag.initial_position, ag.initial_direction, ag.target,
            rail, planned, max_timestep, hd, deadline=deadlines[agent_id],
        )
        path_all[agent_id] = path
        planned.append(path)
    return path_all


def evaluate_paths(paths, agents, max_timestep, deadlines) -> Tuple[int, int]:
    """Returns (failed_count, sic_plus_penalty)."""
    failed = 0
    cost = 0
    for i, (p, ag) in enumerate(zip(paths, agents)):
        if not p:
            failed += 1
            cost += max_timestep
            continue
        path_cost = len(p) - 1
        ddl = deadlines[i] if i < len(deadlines) else max_timestep
        delay = max(0, path_cost - ddl)
        cost += path_cost + 2 * delay
    return failed, cost


# ════════════════════════════════════════════════════════════════════════════
#  Instance loading
# ════════════════════════════════════════════════════════════════════════════

def load_instance(pkl_path: str, ddl_path: Optional[str] = None):
    import pickle
    import json
    import numpy as np

    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"  [WARN] Cannot load {pkl_path}: {e}")
        return None

    try:
        # ... (same parsing logic as before) ...
        if isinstance(data, dict):
            agents = data.get('agents') or data.get('agent_list') or []
            rail = data.get('rail') or data.get('grid') or data.get('transition_map')
            tmax = data.get('max_timestep') or data.get('T_max') or 300
        elif isinstance(data, (tuple, list)) and len(data) >= 2:
            first = data[0]
            if hasattr(first, 'agents'):
                env = first
                agents = env.agents
                rail = env.rail
                tmax = getattr(env, '_max_episode_steps', 300)
            else:
                agents, rail = data[0], data[1]
                tmax = data[2] if len(data) > 2 else 300
        elif hasattr(data, 'agents'):
            agents = data.agents
            rail = data.rail
            tmax = getattr(data, '_max_episode_steps', 300)
        else:
            print(f"  [WARN] Unrecognised pkl format in {pkl_path}")
            return None

        if agents is None or rail is None:
            return None
    except Exception as e:
        print(f"  [WARN] Parsing failed for {pkl_path}: {e}")
        return None

    # deadlines list – initialise with default (max_timestep)
    deadlines = [int(tmax)] * len(agents)

    # Load deadlines from .ddl sidecar if present
    if ddl_path and os.path.exists(ddl_path):
        try:
            with open(ddl_path, 'r') as f:
                ddl_data = json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            try:
                with open(ddl_path, 'rb') as f:
                    ddl_data = pickle.load(f)
            except pickle.PickleError:
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
            if len(ddl_data) < len(agents):
                print(f"  Warning: only {len(ddl_data)} deadlines for {len(agents)} agents")
        else:
            print(f"  No deadlines loaded from {ddl_path}")

    return agents, rail, int(tmax), deadlines   # <-- now returns 4 items

# ════════════════════════════════════════════════════════════════════════════
#  Analytic ranking — O(n log n), no A* required
# ════════════════════════════════════════════════════════════════════════════

def analytic_rank(agents, h_dists: dict, deadlines: list, max_timestep: int) -> np.ndarray:
    n = len(agents)
    def dist(i):
        pos = agents[i].initial_position
        return h_dists[i].get(pos, max_timestep) if pos is not None else max_timestep
    order = sorted(range(n), key=lambda i: (deadlines[i]-dist(i), deadlines[i], -dist(i)))
    ranks = np.zeros(n)
    for rank, agent_id in enumerate(order):
        ranks[agent_id] = rank / max(1, n - 1)
    return ranks

# ════════════════════════════════════════════════════════════════════════════
#  Training data generation — BFS only, no A*
# ════════════════════════════════════════════════════════════════════════════

def generate_training_data(pkl_files: List[str], ddl_files: List[str], n_seeds: int = 8):
    """n_seeds kept for API compat but unused — labels are analytic."""
    all_X = []
    all_y = []

    for pkl_path, ddl_path in zip(pkl_files, ddl_files):
        print(f"  Processing {os.path.basename(pkl_path)} …", end="", flush=True)
        t0 = time.time()
        result = load_instance(pkl_path, ddl_path)
        if result is None:
            print(" SKIP"); continue
        agents, rail, max_timestep, deadlines = result
        n = len(agents)
        if n == 0:
            print(" SKIP (no agents)"); continue

        # BFS per unique goal
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
            grid_rows = len(rail); grid_cols = len(rail[0]) if grid_rows > 0 else 50
        else:
            grid_rows, grid_cols = 50, 50

        feats = extract_features(agents, h_dists, max_timestep, grid_rows, grid_cols, deadlines)
        ranks = analytic_rank(agents, h_dists, deadlines, max_timestep)

        all_X.append(feats)
        all_y.append(ranks)
        print(f" OK  ({n} agents, {time.time()-t0:.1f}s)")

    if not all_X:
        print("[WARN] No training data generated!")
        return np.zeros((0, FEATURE_DIM)), np.zeros(0)

    return np.vstack(all_X), np.concatenate(all_y)


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train AgentPriorityNet on Flatland test cases."
    )
    parser.add_argument("--test_dir",   default="multi_test_case/",
                        help="Directory containing multi-agent .pkl files")
    parser.add_argument("--single_dir", default="single_test_case/",
                        help="Directory containing single-agent .pkl files")
    parser.add_argument("--ddl_dir",    default=None,
                        help="Directory for .ddl deadline files (default: same as test_dir)")
    parser.add_argument("--output",     default="model_weights.npz",
                        help="Where to save trained weights")
    parser.add_argument("--epochs",     type=int, default=150,
                        help="Training epochs")
    parser.add_argument("--lr",         type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--seeds",      type=int, default=10,
                        help="Number of priority orderings to try per instance")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Minibatch size for training")
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
                ddl_candidate = os.path.join(ddl_dir,
                                             os.path.basename(f).replace(".pkl", ".ddl"))
                ddl_files.append(ddl_candidate if os.path.exists(ddl_candidate) else "")

    if not pkl_files:
        print("ERROR: No .pkl files found. Check --test_dir and --single_dir.")
        sys.exit(1)

    print(f"Found {len(pkl_files)} instances.")

    # ── Generate training data ────────────────────────────────────────────
    print("\n=== Generating training data ===")
    X, y = generate_training_data(pkl_files, ddl_files, n_seeds=args.seeds)
    print(f"\nTotal training samples: {X.shape[0]}")

    if X.shape[0] == 0:
        print("No training data — saving default weights.")
        net = AgentPriorityNet()
        net.save(args.output)
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
            Xb = X[batch_idx]
            yb = y[batch_idx]
            # Use pairwise loss for small batches (≤50), MSE for large
            if len(batch_idx) <= 50:
                loss = net.pairwise_train_step(Xb, yb, lr=args.lr)
            else:
                loss = net.train_step(Xb, yb, lr=args.lr)
            epoch_loss += loss
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        if epoch % 10 == 0 or epoch == 1:
            # Compute ordering accuracy: how often does NN give lower score
            # to higher-priority agent?  (for a random sample of 200 pairs)
            scores = net.forward(X)
            correct = 0
            total = 0
            sample_pairs = min(200, n_total * (n_total - 1) // 2)
            for _ in range(sample_pairs):
                i, j = rng.choice(n_total, 2, replace=False)
                if abs(y[i] - y[j]) > 0.05:  # only non-tied pairs
                    correct += int((scores[i] < scores[j]) == (y[i] < y[j]))
                    total += 1
            acc = correct / max(1, total)
            print(f"  Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  "
                  f"pair-acc={acc:.3f}")

    # ── Save ─────────────────────────────────────────────────────────────
    net.save(args.output)
    print(f"\nDone. Model saved to {args.output}")


if __name__ == "__main__":
    main()