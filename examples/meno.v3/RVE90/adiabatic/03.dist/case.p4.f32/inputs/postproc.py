# Plotting quantities
# =====================================
_indices = list(range(45))
_common = {
  "indices":  ({3: 0}, {0: _indices}, {0: _indices}),
  "varnames": None,
  "labels":   [r"$t$ [s]", r"$n$ [m$^{-3}$]"]
}
figures = [
  {
    "figname": "dist_lin",
    "scales":  ["log", "linear"],
    **_common
  },
  {
    "figname": "dist_log",
    "scales":  ["log", "log"],
    **_common
  }
]
