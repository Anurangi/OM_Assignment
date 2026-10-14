"""
Heuristic methods for the p-median problem.

1. Greedy (add)       - constructive baseline: open bases one at a time, each time
                        choosing the site that reduces the total cost the most.
2. Simulated Annealing - starts from the greedy answer and explores "swap" moves:
                        close one open base and open one closed site.
"""
import math
import time
import numpy as np
from .data import Instance


def cost_of(inst: Instance, open_sites) -> float:
    return float(inst.w @ inst.t[:, list(open_sites)].min(axis=1))


# ---------------------------------------------------------------------------
# 1. Greedy add
# ---------------------------------------------------------------------------
def greedy(inst: Instance) -> dict:
    t0 = time.perf_counter()
    best = np.full(inst.I, np.inf)          # current response time of each GN
    open_sites = []
    for _ in range(inst.p):
        # cost if we add site j = sum_i w_i * min(best_i, t_ij), for every j at once
        cand = inst.w @ np.minimum(best[:, None], inst.t)
        cand[open_sites] = np.inf
        j = int(np.argmin(cand))
        open_sites.append(j)
        best = np.minimum(best, inst.t[:, j])
    return {"method": "Greedy", "open": sorted(open_sites),
            "cost": float(inst.w @ best), "runtime": time.perf_counter() - t0}

# ---------------------------------------------------------------------------
# 2. Simulated Annealing
# ---------------------------------------------------------------------------
def _nearest_two(inst, open_arr):
    """For every demand point: time to nearest open base, its position, and time to 2nd."""
    sub = inst.t[:, open_arr]
    if len(open_arr) == 1:
        return sub[:, 0], np.zeros(inst.I, int), np.full(inst.I, np.inf)
    part = np.argpartition(sub, 1, axis=1)[:, :2]
    d0 = np.take_along_axis(sub, part[:, :1], 1)[:, 0]
    d1 = np.take_along_axis(sub, part[:, 1:2], 1)[:, 0]
    swap = d1 < d0
    k0 = np.where(swap, part[:, 1], part[:, 0])
    return np.minimum(d0, d1), k0, np.maximum(d0, d1)


def simulated_annealing(inst: Instance, init_open, seed: int = 0, iters: int = 20000,
                        T0_frac: float = 0.01, T_end_frac: float = 1e-5) -> dict:
    """
    Solution    : a set of p open sites.
    Move        : swap - remove a random open base r, add a random closed site a.
    Delta cost  : computed in O(|I|) using each demand point's nearest and
                  second-nearest open base (no full recomputation needed).
    Acceptance  : Metropolis - accept worse moves with prob exp(-delta/T).
    Temperature : T0 = T0_frac x initial cost, cooled geometrically to
                  T_end_frac x initial cost over `iters` moves.
    Returns the best solution found.
    """
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    open_arr = np.array(sorted(init_open))
    is_open = np.zeros(inst.J, bool); is_open[open_arr] = True
    d0, k0, d1 = _nearest_two(inst, open_arr)
    cur = float(inst.w @ d0)
    best_cost, best_open = cur, open_arr.copy()
    T = T0_frac * cur
    cool = (T_end_frac / T0_frac) ** (1 / iters)
    history, accepted = [(0, cur)], 0

    if inst.J == inst.p:            # nothing to swap
        iters = 0
    for k in range(1, iters + 1):
        pos = rng.integers(inst.p)                       # which open base to close
        closed = np.flatnonzero(~is_open)
        a = closed[rng.integers(len(closed))]            # which site to open
        # new time for each GN: if its nearest base is being closed it falls back
        # to its 2nd nearest; either way it may now use the new site a.
        base = np.where(k0 == pos, d1, d0)
        new_d = np.minimum(base, inst.t[:, a])
        new = float(inst.w @ new_d)
        delta = new - cur
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            is_open[open_arr[pos]] = False; is_open[a] = True
            open_arr[pos] = a
            d0, k0, d1 = _nearest_two(inst, open_arr)
            cur = new; accepted += 1
            if cur < best_cost - 1e-9:
                best_cost, best_open = cur, open_arr.copy()
                history.append((k, cur))
        T *= cool

    return {"method": "SA", "open": sorted(best_open.tolist()), "cost": best_cost,
            "runtime": time.perf_counter() - t0, "iters": iters,
            "accept_rate": accepted / max(iters, 1), "history": history}

