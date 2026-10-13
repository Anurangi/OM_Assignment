"""
Runs every experiment and writes CSV files to results/.

  E1  Realistic case  - candidate bases = DS-division centres; Puttalam, Kurunegala, whole NWP
  E2  Scalability     - whole NWP, candidate bases = 25 ... 2158 GN centres, p = 10
  E3  Policy trade-off- whole NWP, DS centres, p = 4 ... 20 bases

Usage:  python experiments.py            (all, ~30-40 min, mostly ILP time)
        python experiments.py E1 E3      (only some)
"""
import os
import sys
import json
import time
import signal
import multiprocessing as mp
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from ambupos import (make_instance, greedy, simulated_annealing, solve_ilp,
                     solve_lp_relaxation, evaluate)

SA_SEEDS = range(10)
ILP_TIME_LIMIT = 300            # seconds, given to CBC
ILP_HARD_LIMIT = 600            # seconds, wall clock; CBC ignores its own limit while
                                # it is still solving the root LP of a very large model
RESULTS = "results"


def _ilp_worker(inst, q):
    os.setsid()                 # own process group, so the CBC child can be killed too
    q.put(solve_ilp(inst, time_limit=ILP_TIME_LIMIT))


def solve_ilp_hard_limit(inst):
    """Run the ILP in a child process and kill it (and CBC) after ILP_HARD_LIMIT seconds."""
    q = mp.Queue()
    proc = mp.Process(target=_ilp_worker, args=(inst, q))
    t0 = time.perf_counter()
    proc.start()
    proc.join(ILP_HARD_LIMIT)
    if proc.is_alive():
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.join()
        return {"method": "ILP", "open": None, "optimal": False,
                "runtime": time.perf_counter() - t0, "status": "killed: no solution in hard limit",
                "n_vars": inst.J + inst.I * inst.J}
    return q.get()


def run_all_methods(inst, run_ilp=True, run_lp=True, sa_seeds=SA_SEEDS):
    rows, extra = [], {}
    base = {"instance": inst.name, "I": inst.I, "J": inst.J, "p": inst.p}

    g = greedy(inst)
    rows.append({**base, "method": "Greedy", "seed": None, "runtime": g["runtime"],
                 "optimal": None, **evaluate(inst, g["open"]), "open": json.dumps(g["open"])})

    for s in sa_seeds:
        r = simulated_annealing(inst, g["open"], seed=s)
        rows.append({**base, "method": "SA", "seed": s, "runtime": r["runtime"] + g["runtime"],
                     "optimal": None, **evaluate(inst, r["open"]), "open": json.dumps(r["open"])})
        if s == 0:
            extra["sa_history"] = r["history"]

    if run_ilp:
        r = solve_ilp_hard_limit(inst)
        row = {**base, "method": "ILP", "seed": None, "runtime": r["runtime"],
               "optimal": r["optimal"], "n_vars": r["n_vars"], "status": r["status"]}
        if r["open"] is not None:
            row.update({**evaluate(inst, r["open"]), "open": json.dumps(r["open"])})
        rows.append(row)
    if run_lp:
        lp = solve_lp_relaxation(inst)
        extra["lp_bound"] = lp["lp_bound"]
        extra["lp_runtime"] = lp["runtime"]
    return rows, extra


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def e1():
    rows, bounds = [], []
    for district, p in [("Puttalam", 4), ("Kurunegala", 7), (None, 10)]:
        inst = make_instance(p=p, district=district)
        log("E1", inst.name)
        r, extra = run_all_methods(inst)
        rows += r
        bounds.append({"instance": inst.name, "lp_bound": extra["lp_bound"],
                       "lp_runtime": extra["lp_runtime"]})
        if district is None:
            pd.DataFrame(extra["sa_history"], columns=["iter", "cost"]).to_csv(
                f"{RESULTS}/e1_sa_history.csv", index=False)
    pd.DataFrame(rows).to_csv(f"{RESULTS}/e1_realistic.csv", index=False)
    pd.DataFrame(bounds).to_csv(f"{RESULTS}/e1_lp_bounds.csv", index=False)


def e2(resume=True):
    path = f"{RESULTS}/e2_scalability.csv"
    old = pd.read_csv(path) if resume and os.path.exists(path) else pd.DataFrame()
    done = set(old.J) if len(old) else set()
    rows = old.to_dict("records")
    for J in [25, 50, 100, 200, 400, 800, 2158]:
        if J in done:
            continue
        inst = make_instance(p=10, sites="gnd", n_sites=J, seed=1)
        log("E2", inst.name)
        r, _ = run_all_methods(inst, run_ilp=J <= 400, run_lp=False)
        rows += r
        pd.DataFrame(rows).to_csv(path, index=False)


def e3():
    rows = []
    for p in [4, 6, 8, 10, 12, 15, 20]:
        inst = make_instance(p=p)
        log("E3", inst.name)
        r, _ = run_all_methods(inst, run_ilp=True, run_lp=False, sa_seeds=range(5))
        rows += r
        pd.DataFrame(rows).to_csv(f"{RESULTS}/e3_tradeoff.csv", index=False)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["E1", "E2", "E3"]
    for name in todo:
        {"E1": e1, "E2": e2, "E3": e3}[name]()
    log("done")
