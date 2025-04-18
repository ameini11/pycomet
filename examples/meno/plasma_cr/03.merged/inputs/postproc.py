# Plotting quantities
# =====================================
_indices = list(range(45))
_common = {
  "indices":  ({3: 0}, {1: _indices}, {1: _indices}),
  "varnames": None,
  "labels":   [r"$t$ [s]", r"$n$ [m$^{-3}$]"]
}
figures = [
  {
    "figname":  "mass",
    "indices":  ({3: 0}, {0: 0}, {0: 0}),
    "varnames": [r"O"],
    "labels":   [r"$t$ [s]", r"$n$ [m$^{-3}$]"],
    "scales":   ["log", "linear"]
  },
  {
    "figname": "dist_lin",
    "scales":  ["log", "linear"],
    **_common
  },
  {
    "figname": "dist_log",
    "scales":  ["log", "log"],
    **_common
  },
  {
    "figname":  "temp",
    "indices":  ({3: 0}, {2: 0}, {2: 0}),
    "varnames": None,
    "labels":   [r"$t$ [s]", r"$T$ [K]"],
    "scales":   ["log", "linear"]
  }
]
