"""
Data for the ambulance base location problem in the North Western Province (NWP).

Real data (Census of Population and Housing 2012, Dept. of Census & Statistics,
republished at github.com/nuuuwan/gig-data):
  data/raw/nwp_gnd.csv - 2,158 Grama Niladhari divisions: name, centroid, population
  data/raw/nwp_dsd.csv -    46 Divisional Secretariat divisions: name, centroid, population

Modelled part (stated assumption):
  travel time = road distance / average speed, where
  road distance = straight-line (haversine) distance x CIRCUITY.
"""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

CIRCUITY = 1.4          # roads are ~40% longer than a straight line (rural Sri Lanka)
SPEED_KMH = 40.0        # average ambulance speed on mixed rural/town roads
EARTH_R_KM = 6371.0


@dataclass
class Instance:
    name: str
    demand: pd.DataFrame        # one row per demand point (GN division)
    sites: pd.DataFrame         # one row per candidate ambulance base
    p: int                      # number of bases to open
    w: np.ndarray = field(init=False)   # population weights, shape (I,)
    t: np.ndarray = field(init=False)   # travel time in minutes, shape (I, J)

    def __post_init__(self):
        self.w = self.demand["population"].to_numpy(dtype=float)
        self.t = travel_time_matrix(self.demand, self.sites)

    @property
    def I(self):            # number of demand points
        return len(self.demand)

    @property
    def J(self):            # number of candidate sites
        return len(self.sites)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Accepts numpy arrays (broadcasts)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def travel_time_matrix(demand: pd.DataFrame, sites: pd.DataFrame) -> np.ndarray:
    """t[i, j] = minutes for an ambulance at site j to reach demand point i."""
    d = haversine_km(demand["lat"].to_numpy()[:, None], demand["lon"].to_numpy()[:, None],
                     sites["lat"].to_numpy()[None, :], sites["lon"].to_numpy()[None, :])
    return d * CIRCUITY / SPEED_KMH * 60.0


def load_gnd(district: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "nwp_gnd.csv")
    return df if district is None else df[df["district"] == district].reset_index(drop=True)


def load_dsd(district: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "nwp_dsd.csv")
    return df if district is None else df[df["district"] == district].reset_index(drop=True)


def make_instance(p: int, district: str | None = None, sites: str = "dsd",
                  n_sites: int | None = None, seed: int = 0) -> Instance:
    """Build a problem instance.

    district : "Puttalam", "Kurunegala" or None (whole province)
    sites    : "dsd" -> the DS-division centres are the candidate bases (realistic:
                        each DS centre has a town, hospital or police station)
               "gnd" -> n_sites GN-division centres sampled at random, population-
                        weighted (used to scale the problem up for experiments)
    """
    demand = load_gnd(district)
    if sites == "dsd":
        cand = load_dsd(district)
    else:
        rng = np.random.default_rng(seed)
        prob = demand["population"] / demand["population"].sum()
        k = min(n_sites, len(demand))
        idx = rng.choice(len(demand), size=k, replace=False, p=prob)
        cand = demand.iloc[np.sort(idx)].reset_index(drop=True)
    tag = district or "NWP"
    name = f"{tag}_{sites}{'' if sites == 'dsd' else n_sites}_p{p}"
    return Instance(name, demand, cand.reset_index(drop=True), p)
