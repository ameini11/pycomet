# Plotting quantities
# =====================================
_indices = list(range(34))
_common = {
  "indices":  ({3: 0}, {0: _indices}, {0: _indices}),
  "varnames": None,
  "labels":   [r"$t$ [s]", r"$w$"]
}
figures = [
  {
    "figname": "states_lin",
    "scales":  ["log", "linear"],
    **_common
  },
  {
    "figname": "states_log",
    "scales":  ["log", "log"],
    **_common
  },
  {
    "figname":  "Th",
    "indices":  ({3: 0}, {1: 0}, {1: 0}),
    "varnames": None,
    "labels":   [r"$t$ [s]", r"$T_h$ [K]"],
    "scales":   ["log", "linear"]
  },
  {
    "figname":  "Te",
    "indices":  ({3: 0}, {2: 0}, {2: 0}),
    "varnames": None,
    "labels":   [r"$t$ [s]", r"$T_e$ [K]"],
    "scales":   ["log", "linear"]
  }
]
