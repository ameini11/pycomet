from inputs.data import _columns

# Plotting quantities
# =====================================
_nb_vars = len(_columns["y"][0])
_indices = list(range(_nb_vars))
_common = {
  "indices": ({3: 0}, {0: _indices}, {0: _indices}),
  "labels":  [r"$t$ [min]", r"$y$ [ppm]"]
}
figures = [
  {
    "figname": "y_lin_lin",
    "scales":  ["log", "linear"],
    **_common
  },
  {
    "figname": "y_lin_log",
    "scales":  ["log", "log"],
    "ylim":    [1e-20, 1e1],
    **_common
  }
]
