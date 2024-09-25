# Plotting quantities
# =====================================
_common = {
  "indices":  ({3: 0}, {0: 0}, {0: 0}),
  "varnames": [r"OH"],
  "labels":   [r"$t$ [min]", r"$y$ [ppm]"]
}
figures = [
  {
    "figname":  "y6_lin",
    "scales":   ["log", "linear"],
    **_common
  },
  {
    "figname":  "y6_log",
    "scales":   ["log", "log"],
    **_common
  }
]
