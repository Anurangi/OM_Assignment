"""Makes every figure in figures/ from the CSV files in results/."""
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
from ambupos import make_instance, assign_nearest, load_dsd

plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3})
C = {"ILP": "#1f4e79", "SA": "#d1495b", "Greedy": "#8c8c8c"}
R, F = "results", "figures"


def fig_map():
    """Map of the optimal 10 bases (DS centres) and which GN divisions each one serves."""
    df = pd.read_csv(f"{R}/e1_realistic.csv")
    row = df[(df.instance == "NWP_dsd_p10") & (df.method == "ILP")].iloc[0]
    inst = make_instance(p=10)
    open_ = json.loads(row["open"])
    serve = assign_nearest(inst, open_)
    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    cmap = plt.get_cmap("tab10")
    for k, j in enumerate(open_):
        m = serve == j
        ax.scatter(inst.demand.lon[m], inst.demand.lat[m], s=np.sqrt(inst.w[m]) / 6,
                   color=cmap(k), alpha=0.55, lw=0)
    s = inst.sites
    ax.scatter(s.lon, s.lat, marker="x", color="k", s=18, lw=0.8, label="Candidate site (DS centre)")
    ax.scatter(s.lon[open_], s.lat[open_], marker="*", s=260, color="gold", edgecolor="k",
               zorder=5, label="Chosen ambulance base")
    for j in open_:
        ax.annotate(s.name[j], (s.lon[j], s.lat[j]), xytext=(5, 4), textcoords="offset points",
                    fontsize=7.5, weight="bold")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"Optimal 10 ambulance bases, North Western Province\n"
                 f"(dot = GN division, size ~ population, colour = serving base)\n"
                 f"avg response {row.avg_time:.1f} min, {row.cover_15:.0%} of people within 15 min",
                 fontsize=9.5)
    ax.set_aspect(1 / np.cos(np.radians(7.8)))
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{F}/fig1_map_optimal_bases.png"); plt.close(fig)


def fig_scalability():
    df = pd.read_csv(f"{R}/e2_scalability.csv")
    agg = df.groupby(["J", "method"]).agg(runtime=("runtime", "mean"),
                                          avg_time=("avg_time", "min")).reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in ["ILP", "SA", "Greedy"]:
        a = agg[agg.method == m]
        ax.plot(a.J, a.runtime, "o-", color=C[m],
                label={"SA": "SA (incl. greedy start, mean of 10 seeds)"}.get(m, m))
    ilp = df[df.method == "ILP"]
    hit = ilp[(ilp.optimal == False) & ilp.objective.notna()]
    if len(hit):
        ax.scatter(hit.J, hit.runtime, s=130, facecolor="none", edgecolor="k", zorder=5,
                   label="ILP stopped at time limit (best found, not proven optimal)")
    dead = ilp[ilp.objective.isna()]
    if len(dead):
        ax.scatter(dead.J, dead.runtime, marker="X", s=110, color="k", zorder=6,
                   label="ILP killed after 10 min with no solution")
    ax.axhline(600, color="k", lw=0.8, ls=":")
    ax.text(26, 640, "10-minute hard limit", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of candidate sites |J|  (|I| = 2,158 GN divisions, p = 10)")
    ax.set_ylabel("Runtime (seconds, log scale)")
    ax.set_title("Runtime vs problem size")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{F}/fig2_runtime_scalability.png"); plt.close(fig)

    # quality gap vs best known solution for each size
    df = df[df.objective.notna()]
    best = df.groupby("J").objective.min().rename("best")
    df = df.join(best, on="J")
    df["gap_pct"] = 100 * (df.objective - df.best) / df.best
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in ["Greedy", "SA", "ILP"]:
        a = df[df.method == m].groupby("J").gap_pct
        mean, lo, hi = a.mean(), a.min(), a.max()
        ax.plot(mean.index, mean.values, "o-", color=C[m], label=m)
        if m == "SA":
            ax.fill_between(mean.index, lo.values, hi.values, color=C[m], alpha=0.2,
                            label="SA range over 10 seeds")
    ax.set_xscale("log")
    ax.set_xlabel("Number of candidate sites |J|")
    ax.set_ylabel("Gap to best known solution (%)")
    ax.set_title("Solution quality vs problem size")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{F}/fig3_quality_gap.png"); plt.close(fig)


def fig_tradeoff():
    df = pd.read_csv(f"{R}/e3_tradeoff.csv")
    ilp = df[df.method == "ILP"].sort_values("p")
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax1.plot(ilp.p, ilp.avg_time, "o-", color=C["ILP"], label="Average response time")
    ax1.set_xlabel("Number of ambulance bases p")
    ax1.set_ylabel("Average response time (min)", color=C["ILP"])
    ax2 = ax1.twinx()
    ax2.plot(ilp.p, 100 * ilp.cover_15, "s--", color="#2a9d8f", label="Within 15 min")
    ax2.plot(ilp.p, 100 * ilp.cover_30, "^--", color="#e9c46a", label="Within 30 min")
    ax2.set_ylabel("Population covered (%)")
    ax2.grid(False); ax2.spines["right"].set_visible(True)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    ax1.set_title("How many bases does the province need? (optimal ILP solutions)")
    fig.tight_layout(); fig.savefig(f"{F}/fig4_bases_tradeoff.png"); plt.close(fig)


def fig_sa_convergence():
    h = pd.read_csv(f"{R}/e1_sa_history.csv")
    e1 = pd.read_csv(f"{R}/e1_realistic.csv")
    W = make_instance(p=10).w.sum()
    opt = e1[(e1.instance == "NWP_dsd_p10") & (e1.method == "ILP")].objective.iloc[0]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = list(h["iter"]) + [20000]
    y = list(h["cost"] / W) + [h["cost"].iloc[-1] / W]
    ax.step(x, y, where="post", color=C["SA"], label="SA best-so-far (seed 0)")
    ax.axhline(opt / W, color=C["ILP"], ls="--", label="ILP optimum")
    ax.axhline(h["cost"].iloc[0] / W, color=C["Greedy"], ls=":", label="Greedy start")
    ax.set_xlabel("SA iteration"); ax.set_ylabel("Average response time (min)")
    ax.set_title("Simulated Annealing convergence (NWP, 46 DS sites, p = 10)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{F}/fig5_sa_convergence.png"); plt.close(fig)


if __name__ == "__main__":
    for f in [fig_map, fig_scalability, fig_tradeoff, fig_sa_convergence]:
        try:
            f(); print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, "-", e)
