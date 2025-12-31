from inputs.data import species_indices

# Plotting quantities
# =====================================
_indices_lin, _indices_nlin = [  
  list(range(len(species_indices[k]))) for k in ("lin", "nlin")
]
figures = [
  {
    "figname": "y_lin_lin",
    "scales":  ["log", "linear"],
    "indices": ({3: 0}, {0: _indices_lin}, {0: _indices_lin}),
    "labels":  [r"$t$ [min]", r"$y$ [ppm]"]
  },
  {
    "figname": "y_lin_log",
    "scales":  ["log", "log"],
    "ylim":    [1e-20, 1e1],
    "indices": ({3: 0}, {0: _indices_lin}, {0: _indices_lin}),
    "labels":  [r"$t$ [min]", r"$y$ [ppm]"]
  },
  {
    "figname": "y_nlin_lin",
    "scales":  ["log", "linear"],
    "indices": ({3: 0}, {1: _indices_nlin}, {1: _indices_nlin}),
    "labels":  [r"$t$ [min]", r"$y$ [ppm]"]
  },
  {
    "figname": "y_nlin_log",
    "scales":  ["log", "log"],
    "ylim":    [1e-20, 1e1],
    "indices": ({3: 0}, {1: _indices_nlin}, {1: _indices_nlin}),
    "labels":  [r"$t$ [min]", r"$y$ [ppm]"]
  }
]
