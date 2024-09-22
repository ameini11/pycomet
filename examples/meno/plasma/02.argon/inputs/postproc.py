# Plotting quantities
# =====================================
_indices = list(range(4))
_common = {
  "indices":  ({3: 0}, {0: _indices}, {0: _indices}),
  "varnames": ["Ar", r"Ar$^{4s}$", r"Ar$^{4p}$", r"Ar$^+$"],
  "labels":   [r"$t$ [s]", r"$n$ [m$^{-3}$]"]
}
figures = [
  {
    "figname": "argon_lin",
    "scales":  ["log", "linear"],
    **_common
  },
  {
    "figname": "argon_log",
    "scales":  ["log", "log"],
    "ylim":    [1e10, 5e20],
    **_common
  }
]
