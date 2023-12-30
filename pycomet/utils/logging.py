import os
import numpy as np
import tensorflow as tf

from pycomet.config import get_config
_CFG = get_config()


# Module global variables
# =====================================
EPILOG = "\n" + '='*_CFG.display_len + "\n "
TABS = np.arange(5)*min(_CFG.tab, 2)
HEADERS = {
  "main":  '['+_CFG.name+"]: ",
  "warning": '['+_CFG.name+"]:   WARNING! ",
  "err":   "from "+_CFG.name+"\n>>> "
}

# Printing
# =====================================
def print_fn(
  text,
  start='\n',
  lev=0,
  verbose=True,
  end='\n'
):
  if verbose:
    if lev > 0:
      start=''
    print(start + HEADERS['main'] + ' '*TABS[lev] + text, end=end)

def input_fn(
  text,
  start='',
  lev=0
):
  return input(start + HEADERS['main'] + ' '*TABS[lev] + text)

def warning(
  text,
  *args,
  **kwargs
):
  print_fn(text="WARNING! " + text, *args, **kwargs)

def raise_err(
  error,
  text
):
  message = ''.join([HEADERS['err'], text])
  raise error(message)

# Environment
# =====================================
def environment():
  nb_gpus = len(tf.config.list_physical_devices("GPU"))
  nb_cpus = len(tf.config.list_physical_devices("CPU"))
  delimiter = '-' * _CFG.display_len
  text = HEADERS["main"] + "Environment setup" + \
    "\n" + delimiter + \
		"\n > TensorFlow version:          {}".format(tf.__version__) + \
    "\n > Nb. of CPUs available:       {}".format(nb_cpus) + \
    "\n   >> Nb. of threads available: {}".format(os.cpu_count()) + \
    "\n   >> Nb. of threads used:      {}".format(_CFG.cpu_threads) + \
    "\n > Nb. of GPUs available:       {}".format(nb_gpus) + \
    "\n   >> GPU memory growth:        {}".format(_CFG.gpu_memory_growth) + \
    "\n" + delimiter
  print(text)

# Package header
# =====================================
def pkg_header():
  info = ''.join([
    "\n" + _compose(
      text=line+_compute_blanks(line), start="# "
    ) for line in _CFG.descr_lines
  ])
  text = "\n" + _compose(start="# ", between="*") + \
    "\n" + _compose(
      text=_CFG.name+_compute_blanks(_CFG.name), start="# "
    ) + \
    "\n" + _compose(start="# ", between="-") + info + \
    "\n" + _compose(start="# ") + \
    "\n" + _compose(start="# ", between="*") + "\n"
  print(text)

def _compose(
	text='',
  start='',
  between=' '
):
  l = len(text)
  ftext = start
  ftext += between*int((_CFG.display_len-len(start)*2-l)/2)
  ftext += text
  ftext += between*int((_CFG.display_len-len(start)*2-l)/2)
  ftext += ''.join(reversed(start))
  return ftext

def _compute_blanks(
  text
):
  return ' '*((_CFG.display_len-len(text))%2)
