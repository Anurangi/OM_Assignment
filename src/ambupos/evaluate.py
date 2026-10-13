"""Score any solution (a set of open sites) the same way, whatever method found it."""
import numpy as np
from .data import Instance


def assign_nearest(inst: Instance, open_sites) -> np.ndarray:
    """Each demand point is served by its nearest open base."""
    open_sites = np.asarray(open_sites)
    return open_sites[np.argmin(inst.t[:, open_sites], axis=1)]


def evaluate(inst: Instance, open_sites, thresholds=(15, 30)) -> dict:
    open_sites = np.asarray(sorted(open_sites))
    tt = inst.t[:, open_sites].min(axis=1)          # each GN's response time
    W = inst.w.sum()
    out = {
        "feasible": len(set(open_sites.tolist())) == inst.p,
        "objective": float(inst.w @ tt),             # population-weighted minutes
        "avg_time": float(inst.w @ tt / W),          # minutes for the average person
        "max_time": float(tt.max()),                 # worst-served GN division
    }
    for m in thresholds:
        out[f"cover_{m}"] = float(inst.w[tt <= m].sum() / W)   # share of people within m min
    return out
