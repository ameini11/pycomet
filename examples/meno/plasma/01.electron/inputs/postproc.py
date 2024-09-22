# Plotting quantities
# =====================================
_common = {
  "indices":  ({3: 0}, {0: 0}, {0: 0}),
  "varnames": [r"e$^{-}$"],
  "labels":   [r"$t$ [s]", r"$n$ [m$^{-3}$]"]
}
figures = [
  {
    "figname":  "electron_lin",
    "scales":   ["log", "linear"],
    **_common
  },
  {
    "figname":  "electron_log",
    "scales":   ["log", "log"],
    **_common
  }
]
