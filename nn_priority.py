"""
nn_priority.py
==============
Lightweight numpy-only MLP for Flatland agent priority prediction.

Architecture:  8 → 32 → 16 → 1  (wider than before, same depth)
Optimiser:     Adam with gradient clipping (replaces raw SGD — fixes overflow)
Init:          He normal (prevents vanishing/exploding gradients from the start)
NaN guard:     forward() falls back to analytic slack score if weights are bad

Features (index, name, description):
  0  dist_norm          BFS distance to goal / max_timestep
  1  deadline_norm      deadline / max_timestep
  2  slack_norm         (deadline - dist) / max_timestep   ← most informative
  3  n_agents_norm      num_agents / 200
  4  start_x_norm       agent row / grid_rows
  5  start_y_norm       agent col / grid_cols
  6  congestion         fraction of other agents within 10 BFS steps
  7  dist_nearest_norm  BFS dist to nearest other agent's start / 50
"""

import numpy as np
import os
from typing import Optional, List

FEATURE_DIM = 8


# ─────────────────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────────────────
class AgentPriorityNet:
    """
    3-layer MLP (8 → 32 → 16 → 1) with:
      • He normal initialisation  — no more overflow on first forward pass
      • Adam optimiser            — adaptive lr, much faster convergence
      • Gradient clipping (norm)  — prevents explosion even with large batches
      • NaN guard in forward()    — falls back to analytic score if weights bad
    """

    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)

        # He normal: std = sqrt(2 / fan_in)
        self.W1 = rng.standard_normal((FEATURE_DIM, 32)) * np.sqrt(2.0 / FEATURE_DIM)
        self.b1 = np.zeros(32)

        self.W2 = rng.standard_normal((32, 16)) * np.sqrt(2.0 / 32)
        self.b2 = np.zeros(16)

        self.W3 = rng.standard_normal((16, 1)) * np.sqrt(2.0 / 16)
        self.b3 = np.zeros(1)

        # Bias W1 toward the most informative features AFTER He scaling
        # so the prior is correct but magnitudes are still controlled
        self.W1[2, :16] += 0.3   # slack_norm  → first half of hidden
        self.W1[1, 16:] += 0.2   # deadline    → second half
        self.W1[0, :8]  += 0.15  # dist_norm
        self.W1[6, 8:16]+= 0.1   # congestion

        # Adam moment buffers — initialised lazily on first train_step call
        self._adam: dict = {}
        self._t: int = 0          # global step counter
    
    def _print_weight_stats(self, tag: str, show_full: bool = False):
        """Print min, max, mean, std, and fraction of non‑finite values for all weights."""
        for name in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
            w = getattr(self, name)
            fin_frac = np.mean(np.isfinite(w))
            if fin_frac < 1.0:
                print(f"[{tag}] {name}: non-finite fraction = {1-fin_frac:.2%}")
            else:
                print(f"[{tag}] {name}: min={w.min():8.4f}  max={w.max():8.4f}  "
                    f"mean={w.mean():8.4f}  std={w.std():8.4f}")
            if show_full:
                print(f"     {name} = {w}")

    # ── Activations ──────────────────────────────────────────────────────
    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    # ── Forward pass ─────────────────────────────────────────────────────
    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=2.0, neginf=-2.0)
        X = np.clip(X, -2.0, 2.0)

        self._print_weight_stats("forward (pre)")

        # Guard against exploded or non-finite weights before the matmul.
        # Magnitude cap (<=100) matches the threshold used in the train functions.
        def _weights_ok(w):
            return np.all(np.isfinite(w)) and np.abs(w).max() <= 100.0
        if not (_weights_ok(self.W1) and _weights_ok(self.W2) and _weights_ok(self.W3)):
            return X[:, 2]   # fallback to analytic slack
        z1 = X @ self.W1 + self.b1
        a1 = self._relu(np.clip(z1, -10.0, 10.0))
        z2 = a1 @ self.W2 + self.b2
        a2 = self._relu(np.clip(z2, -10.0, 10.0))
        out = (a2 @ self.W3 + self.b3).squeeze(-1)
        if not np.all(np.isfinite(out)):
            return X[:, 2]   # fallback to analytic slack
        return out

    # ── Ordering ─────────────────────────────────────────────────────────
    def predict_ordering(self, features: np.ndarray) -> list:
        scores = self.forward(features)
        return np.argsort(scores).tolist()

    # ── Adam helper ──────────────────────────────────────────────────────
    def _adam_update(self, param_name: str, param: np.ndarray,
                     grad: np.ndarray, lr: float,
                     beta1: float = 0.9, beta2: float = 0.999,
                     eps: float = 1e-8) -> np.ndarray:
        """In-place Adam update; returns updated parameter."""
        if param_name not in self._adam:
            self._adam[param_name] = {
                'm': np.zeros_like(param),
                'v': np.zeros_like(param),
            }
        buf = self._adam[param_name]
        buf['m'] = beta1 * buf['m'] + (1 - beta1) * grad
        buf['v'] = beta2 * buf['v'] + (1 - beta2) * grad ** 2
        m_hat = buf['m'] / (1 - beta1 ** self._t)
        v_hat = buf['v'] / (1 - beta2 ** self._t)
        return param - lr * m_hat / (np.sqrt(v_hat) + eps)

    # ── Gradient clipping ────────────────────────────────────────────────
    @staticmethod
    def _clip_grads(grads: List[np.ndarray], max_norm: float = 1.0):
        total = np.sqrt(sum(np.sum(g ** 2) for g in grads))
        if total > max_norm:
            scale = max_norm / (total + 1e-8)
            return [g * scale for g in grads]
        return grads

    # ── MSE train step ───────────────────────────────────────────────────
    def train_step(self, X: np.ndarray, y_rank: np.ndarray, lr: float = 1e-3) -> float:
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=2.0, neginf=-2.0)   # same guard as forward()
        X = np.clip(X, -2.0, 2.0)
        n = X.shape[0]
        self._t += 1
        for name, w in [('W1', self.W1), ('W2', self.W2), ('W3', self.W3)]:
            if not np.all(np.isfinite(w)) or np.abs(w).max() > 100.0:
                print(f"[nn_priority] WARNING: {name} exploded — reinitialising")
                self.__init__()
                return 0.0
            
        self._print_weight_stats("train_step (pre forward)")

        # Forward — matches forward() exactly: clip pre-activations, ReLU, no weight clip
        z1 = X @ self.W1 + self.b1
        z1 = np.clip(z1, -10.0, 10.0)
        z1 = np.nan_to_num(z1, nan=0.0, posinf=10.0, neginf=0.0)  # NaN survives clip
        a1 = self._relu(z1)          # a1 in [0, 10]
        z2 = a1 @ self.W2 + self.b2
        z2 = np.clip(z2, -10.0, 10.0)
        z2 = np.nan_to_num(z2, nan=0.0, posinf=10.0, neginf=0.0)  # NaN survives clip
        a2 = self._relu(z2)          # a2 in [0, 10]
        z3 = a2 @ self.W3 + self.b3
        z3 = np.clip(z3, -10.0, 10.0)
        z3 = np.nan_to_num(z3, nan=0.0, posinf=10.0, neginf=0.0)  # NaN survives clip
        pred = z3.squeeze(-1)

        diff = pred - y_rank
        loss = float(np.mean(diff ** 2))
        if not np.isfinite(loss):
            return 0.0

        # Backward — mask accounts for BOTH ReLU (z>0) AND upper clip (z<10)
        # so gradient is zero wherever the clip was saturating at +10.
        dout = (2.0 / n) * diff
        dout3 = dout[:, None]
        dout3 = np.nan_to_num(dout3, nan=0.0, posinf=0.0, neginf=0.0)  # belt-and-braces

        dW3 = a2.T @ dout3
        db3 = dout3.sum(axis=0)
        da2 = dout3 @ self.W3.T

        dz2 = da2 * ((z2 > 0) & (z2 < 10.0))   # correct clip+ReLU mask
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T

        dz1 = da1 * ((z1 > 0) & (z1 < 10.0))   # correct clip+ReLU mask
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)

        grads = [dW1, db1, dW2, db2, dW3, db3]
        dW1, db1, dW2, db2, dW3, db3 = self._clip_grads(grads)

        self.W1 = self._adam_update('W1', self.W1, dW1, lr)
        self.b1 = self._adam_update('b1', self.b1, db1, lr)
        self.W2 = self._adam_update('W2', self.W2, dW2, lr)
        self.b2 = self._adam_update('b2', self.b2, db2, lr)
        self.W3 = self._adam_update('W3', self.W3, dW3, lr)
        self.b3 = self._adam_update('b3', self.b3, db3, lr)

        self._print_weight_stats("train_step (post update)")

        return loss

    # ── Pairwise ranking loss ────────────────────────────────────────────
    def pairwise_train_step(self, X: np.ndarray, ranks: np.ndarray,
                            lr: float = 1e-3, margin: float = 0.1) -> float:
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=2.0, neginf=-2.0)   # same guard as forward()
        X = np.clip(X, -2.0, 2.0)
        n = X.shape[0]
        self._t += 1

        for name, w in [('W1', self.W1), ('W2', self.W2), ('W3', self.W3)]:
            if not np.all(np.isfinite(w)) or np.abs(w).max() > 100.0:
                print(f"[nn_priority] WARNING: {name} exploded — reinitialising")
                self.__init__()
                return 0.0

        # Forward — matches forward() exactly
        z1 = X @ self.W1 + self.b1
        z1 = np.clip(z1, -10.0, 10.0)
        z1 = np.nan_to_num(z1, nan=0.0, posinf=10.0, neginf=0.0)  # NaN survives clip
        a1 = self._relu(z1)          # a1 in [0, 10]
        z2 = a1 @ self.W2 + self.b2
        z2 = np.clip(z2, -10.0, 10.0)
        z2 = np.nan_to_num(z2, nan=0.0, posinf=10.0, neginf=0.0)  # NaN survives clip
        a2 = self._relu(z2)          # a2 in [0, 10]
        z3 = a2 @ self.W3 + self.b3
        z3 = np.clip(z3, -10.0, 10.0)
        z3 = np.nan_to_num(z3, nan=0.0, posinf=10.0, neginf=0.0)  # NaN survives clip
        z3 = z3.squeeze(-1)

        # Vectorised pairwise hinge loss
        score_diff = z3[:, None] - z3[None, :]
        rank_diff = ranks[:, None] - ranks[None, :]
        should_be_first = rank_diff < 0
        hinge = np.maximum(0.0, score_diff + margin) * should_be_first
        n_pairs = should_be_first.sum()
        loss = float(hinge.sum() / max(1, n_pairs))
        if not np.isfinite(loss):
            return 0.0

        # Gradient of hinge w.r.t. z3 scores
        active = (hinge > 0)
        dz3 = (active.sum(axis=1) - active.sum(axis=0)) / max(1, n_pairs)

        # NaN guard on gradients — if dz3 is bad, Adam buffers would be
        # permanently corrupted, so bail out before touching any weights.
        if not np.all(np.isfinite(dz3)):
            return 0.0

        dout3 = dz3[:, None]
        dout3 = np.nan_to_num(dout3, nan=0.0, posinf=0.0, neginf=0.0)  # belt-and-braces

        dW3 = a2.T @ dout3
        db3 = dout3.sum(axis=0)
        da2 = dout3 @ self.W3.T

        dz2 = da2 * ((z2 > 0) & (z2 < 10.0))   # correct clip+ReLU mask
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T

        dz1 = da1 * ((z1 > 0) & (z1 < 10.0))   # correct clip+ReLU mask
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)

        grads = [dW1, db1, dW2, db2, dW3, db3]
        dW1, db1, dW2, db2, dW3, db3 = self._clip_grads(grads)

        self.W1 = self._adam_update('W1', self.W1, dW1, lr)
        self.b1 = self._adam_update('b1', self.b1, db1, lr)
        self.W2 = self._adam_update('W2', self.W2, dW2, lr)
        self.b2 = self._adam_update('b2', self.b2, db2, lr)
        self.W3 = self._adam_update('W3', self.W3, dW3, lr)
        self.b3 = self._adam_update('b3', self.b3, db3, lr)

        self._print_weight_stats("train_step (post update)")

        return loss

    # ── Persistence ──────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        np.savez(path,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3)
        print(f"[nn_priority] Saved model → {path}")

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            d = np.load(path)
            W1, b1 = d['W1'], d['b1']
            W2, b2 = d['W2'], d['b2']
            W3, b3 = d['W3'], d['b3']
            # Reject NaN/Inf weights
            for arr in [W1, W2, W3]:
                if not np.all(np.isfinite(arr)):
                    raise ValueError("NaN/Inf in saved weights")
            # Reject exploded weights (magnitude > 100 means training diverged)
            for name, arr in [('W1', W1), ('W2', W2), ('W3', W3)]:
                mag = np.abs(arr).max()
                if mag > 100.0:
                    raise ValueError(f"{name} magnitude {mag:.2e} too large — retrain")
            self.W1, self.b1 = W1, b1
            self.W2, self.b2 = W2, b2
            self.W3, self.b3 = W3, b3
            print(f"[nn_priority] Loaded model ← {path}")
            return True
        except Exception as e:
            print(f"[nn_priority] Load failed ({e}), using He-initialised weights.")
            self.__init__()   # reinitialise cleanly
            return False


# ─────────────────────────────────────────────────────────────────────────────
#  Feature extraction  (unchanged API)
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(agents, h_dists: dict, max_timestep: int,
                     grid_rows: int = 50, grid_cols: int = 50,
                     deadlines: Optional[list] = None) -> np.ndarray:
    """
    Build the (n_agents, FEATURE_DIM) feature matrix.
    All values normalised to approximately [0, 1].
    """
    n = len(agents)
    X = np.zeros((n, FEATURE_DIM))

    if deadlines is None:
        deadlines = [max_timestep] * n

    starts = [ag.initial_position if ag.initial_position is not None else (0, 0)
              for ag in agents]

    for i, ag in enumerate(agents):
        pos   = ag.initial_position or (0, 0)
        hdist = h_dists.get(i, {})
        dist  = hdist.get(pos, max_timestep)
        ddl   = deadlines[i] if i < len(deadlines) else max_timestep
        slack = ddl - dist

        near = sum(1 for j, s in enumerate(starts)
                   if j != i and hdist.get(s, 999) <= 10)
        congestion = near / max(1, n - 1)

        dists_to_others = [hdist.get(s, 999) for j, s in enumerate(starts) if j != i]
        min_dist = min(dists_to_others, default=50)

        X[i] = [
            min(dist, max_timestep) / max(1, max_timestep),
            min(ddl,  max_timestep) / max(1, max_timestep),
            np.clip(slack, -max_timestep, max_timestep) / max(1, max_timestep),
            n / 200.0,
            pos[0] / max(1, grid_rows),
            pos[1] / max(1, grid_cols),
            congestion,
            min(min_dist, 50) / 50.0,
        ]

    # Clamp any inf/nan that BFS or division may have introduced
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    X = np.clip(X, 0.0, 1.0)   # all features are normalised to [0,1]
    return X