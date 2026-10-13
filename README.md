# Ambulance Base Location – North Western Province, Sri Lanka

Optimization Methods – Programming Assignment

**Problem:** choose where to place *p* ambulance bases so that the population-weighted travel time
to all 2,158 Grama Niladhari (GN) divisions of the province is as small as possible
(the **p-median problem**, NP-hard).

## Methods

- **Exact:** Integer Linear Programming, solved with PuLP + CBC (`src/ambupos/ilp.py`)
- **Heuristic:** Greedy, then Simulated Annealing (`src/ambupos/heuristics.py`)

## Data

Census of Population and Housing 2012 (Department of Census and Statistics), GN- and DS-division
populations and locations, from <https://github.com/nuuuwan/gig-data>. Stored in `data/raw/`.

Travel time is estimated as straight-line distance × 1.4 ÷ 40 km/h.

## How to run

```bash
pip install -r requirements.txt
python experiments.py     # runs all experiments (~40 min), saves results/
python plots.py           # saves figures/
```

Or open `ambulance_location.ipynb` for a walkthrough with a quick demo.

## Main result

For 10 bases across the province, the ILP found the optimal placement (average response time 21.3 min)
in 34 s. Simulated Annealing found the same answer in 0.4 s, and it still worked on large instances
where the ILP could not finish within 10 minutes.

## Team

- Sharala
- Anurangi