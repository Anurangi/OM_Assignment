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


