"""
Exact method: the p-median Integer Linear Program, solved by CBC (branch-and-bound) via PuLP.

Sets        i in I  demand points (GN divisions)        j in J  candidate bases
Parameters  w_i     population of GN division i
            t_ij    travel time (minutes) from base j to GN division i
            p       number of ambulance bases to open
Variables   y_j  in {0,1}   1 if a base is opened at site j
            x_ij in [0,1]   1 if GN division i is served by base j
Model
    min   sum_i sum_j  w_i * t_ij * x_ij          (total population-weighted response time)
    s.t.  sum_j x_ij = 1           for all i      (every GN division is served)
          x_ij <= y_j              for all i, j   (only by an open base)
          sum_j y_j = p                           (exactly p bases)
          y_j binary
x_ij does not need to be declared binary: once y is fixed, sending each i to its
nearest open base is optimal, so an integral x always exists. This keeps the
model to |J| integer variables instead of |I|*|J|.
"""
import time
import numpy as np
import pulp
from .data import Instance


def build_model(inst: Instance, relax: bool = False):
    I, J = range(inst.I), range(inst.J)
    m = pulp.LpProblem("p_median", pulp.LpMinimize)
    y = [pulp.LpVariable(f"y_{j}", 0, 1, cat="Continuous" if relax else "Binary") for j in J]
    x = [[pulp.LpVariable(f"x_{i}_{j}", 0, 1) for j in J] for i in I]

    m += pulp.lpSum(inst.w[i] * inst.t[i, j] * x[i][j] for i in I for j in J)
    for i in I:
        m += pulp.lpSum(x[i][j] for j in J) == 1
        for j in J:
            m += x[i][j] <= y[j]
    m += pulp.lpSum(y) == inst.p
    return m, x, y


def solve_ilp(inst: Instance, time_limit: float = 300, threads: int = 1) -> dict:
    t0 = time.perf_counter()
    m, x, y = build_model(inst)
    t_build = time.perf_counter() - t0
    m.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, threads=threads))
    runtime = time.perf_counter() - t0

    open_sites = [j for j in range(inst.J) if (y[j].value() or 0) > 0.5]
    ok = len(open_sites) == inst.p
    proven = pulp.LpStatus[m.status] == "Optimal" and (runtime - t_build) < 0.98 * time_limit
    return {"method": "ILP", "open": open_sites if ok else None, "optimal": bool(ok and proven),
            "runtime": runtime, "build_time": t_build, "status": pulp.LpStatus[m.status],
            "n_vars": inst.J + inst.I * inst.J, "n_cons": inst.I + inst.I * inst.J + 1}


def solve_lp_relaxation(inst: Instance) -> dict:
    """y continuous: gives a lower bound on the optimal objective."""
    t0 = time.perf_counter()
    m, _, _ = build_model(inst, relax=True)
    m.solve(pulp.PULP_CBC_CMD(msg=0))
    return {"lp_bound": float(pulp.value(m.objective)), "runtime": time.perf_counter() - t0}
