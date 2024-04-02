import numpy as np

# Plotting quantities
# =====================================
_indices = np.arange(60).tolist()
_common = {
  "indices":  ({0: 0}, {0: _indices}, {0: _indices}),
  "varnames": None,
  "labels":   [r"$t$ [s]", r"$n$ [m$^{-3}$]"]
}
figures = [
  {
    "figname": "nm_lin",
    "scales":  ["log", "linear"],
    **_common
  },
  {
    "figname": "nm_log",
    "scales":  ["log", "log"],
    **_common
  }
]
