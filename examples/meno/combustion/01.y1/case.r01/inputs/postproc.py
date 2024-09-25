# Plotting quantities
# =====================================
_common = {
  "indices":  ({3: 0}, {0: 0}, {0: 0}),
  "varnames": [r"NO2"],
  "labels":   [r"$t$ [min]", r"$y$ [ppm]"]
}
figures = [
  {
    "figname":  "y1_lin",
    "scales":   ["log", "linear"],
    **_common
  },
  {
    "figname":  "y1_log",
    "scales":   ["log", "log"],
    **_common
  }
]
