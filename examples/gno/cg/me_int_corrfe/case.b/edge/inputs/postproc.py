# Plotting quantities
# =====================================
_common = {
  "indices":  ({0: 0}, {0: [0,1,2]}, {0: [0,1,2]}),
  "varnames": [r'O$_2^{(%d)}$' % (i+1) for i in range(3)],
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
