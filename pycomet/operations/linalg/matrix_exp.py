"""
Matrix exponential.

The present implementation of the matrix exponential is based on an
optimal Taylor expansion, as detailed in the research paper referenced
https://doi.org/10.3390/math7121174.
"""
import numpy as np
import pycomet as pcm
import tensorflow as tf


@tf.function
def expm(
  A,
  use_pade=False,
  max_degree=None,
  max_nb_squarings=None
):
  A = tf.convert_to_tensor(A)
  if (A.shape[-2:] == [0, 0]):
    return A
  # Reshape in
  A, batch_shape = _reshape_in(A)
  # Compute matrix exponential
  if (A.shape[-1] == 2):
    # > Analytical expression for a 2x2 matrix
    eA = _expm_2by2(A)
  elif use_pade:
    # > Pade approximation
    eA = tf.linalg.expm(A)
  else:
    # > Optimal Taylor expansion
    I = tf.eye(
      num_rows=tf.shape(A)[-2],
      batch_shape=batch_shape,
      dtype=A.dtype
    )
    eA = _taylor(I, A, max_degree, max_nb_squarings)
  # Reshape out
  return _reshape_out(A, eA, batch_shape)

# Reshaping operations
# =====================================
def _reshape_in(A):
  # Batching
  batch_shape = A.shape[:-2]
  if (not batch_shape.is_fully_defined()):
    batch_shape = tf.shape(A)[:-2]
  # Make the tensor 3D: the 'tf.where' statements work better
  A = tf.reshape(
    tensor=A,
    shape=tf.concat(([-1], tf.shape(A)[-2:]), axis=0)
  )
  return A, batch_shape

def _reshape_out(A, eA, batch_shape):
  if (not A.shape.is_fully_defined()):
    return tf.reshape(
      tensor=eA,
      shape=tf.concat((batch_shape, tf.shape(eA)[-2:]), axis=0)
    )
  return tf.reshape(
    tensor=eA,
    shape=batch_shape.concatenate(eA.shape[-2:])
  )

# 2x2 matrix (from SciPy)
# =====================================
def _expm_2by2(A):
  # Split the 4 entries
  A = tf.reshape(A, shape=(-1,4))
  A00, A01, A10, A11 = tf.split(A, num_or_size_splits=4, axis=-1)
  ddiag = A00 - A11
  # Cosh and sinh
  mu = 0.5*tf.math.sqrt(ddiag**2 + 4*A01*A10)
  cosh_mu = tf.math.cosh(mu)
  sinh_mu = tf.ones_like(cosh_mu)
  sinh_mu = tf.where(
    condition=tf.math.not_equal(x=mu, y=tf.constant(0, dtype=mu.dtype)),
    x=tf.math.sinh(mu)/mu,
    y=sinh_mu
  )
  # Compose the 4 entries
  AmD2 = 0.5*ddiag
  eApD2 = tf.math.exp(0.5*(A00+A11))
  eA00 = eApD2 * (cosh_mu + AmD2*sinh_mu)
  eA01 = eApD2 * A01 * sinh_mu
  eA10 = eApD2 * A10 * sinh_mu
  eA11 = eApD2 * (cosh_mu - AmD2*sinh_mu)
  # Concat the 4 entries into a 2x2 matrix
  eA = tf.concat([eA00,eA01,eA10,eA11], axis=-1)
  eA = tf.reshape(eA, shape=(-1,2,2))
  return tf.math.real(eA)

# Optimized Taylor polynomial approximations
# =====================================
def _taylor(I, A, max_degree, max_nb_squarings):
  # Get degrees and norms
  degrees, thetas, max_degree = _get_degree_norm(A, max_degree)
  # Compute L1 norm
  l1_norm = _compute_l1_norm(A)
  # Compute optimized Taylor polynomial approximations
  # > Define stop condition: max polynomial degree reached
  def cond(i, _):
    return tf.math.less_equal(degrees[i], max_degree)
  # > Define Taylor computation
  def body(i, a):
    # Index
    j = i + 1
    # Lower and upper bounds
    lower_bound = tf.math.less_equal(thetas[i], l1_norm)
    upper_bound = tf.math.less(l1_norm, thetas[j])
    # Computations
    e = tf.where(
      tf.math.logical_and(lower_bound, upper_bound),
      x=_taylor_px(I, a, degrees[i], l1_norm, max_nb_squarings),
      y=a
    )
    return j, e
  # > Perform computations
  i = tf.cast(0, dtype=tf.int32)
  _, eA = tf.while_loop(
    cond=cond,
    body=body,
    loop_vars=[i, A]
  )
  return eA

# Polynomial norms and degrees
# -------------------------------------
def _get_degree_norm(A, max_degree):
  dtype = A.dtype
  # Get max norms
  maxnorm = _get_maxnorm(dtype)
  # > Degrees
  degrees = list(maxnorm.keys())
  max_degree_avail = max(degrees)
  degrees += [max_degree_avail+1]
  degrees = tf.constant(degrees, dtype=tf.int32)
  # > Max polynomial degree
  max_degree = max_degree_avail if (max_degree is None) else max_degree
  max_degree = min(max_degree_avail, max_degree)
  # > Norms
  thetas = [0.0] + list(maxnorm.values())
  thetas[-1] = np.inf
  thetas = tf.constant(thetas, dtype=dtype)
  return degrees, thetas, max_degree

def _get_maxnorm(dtype, deg=None):
  if (dtype in (tf.float64, tf.complex128)):
    maxnorm = {
      1:  2.220446049250313e-16,
      2:  2.580956802971767e-08,
      4:  3.397168839976962e-04,
      8:  4.991228871115323e-02,
      12: 2.996158913811580e-01,
      18: 1.090863719290036e+00
    }
  elif (dtype in (tf.float16, tf.float32, tf.complex64)):
    maxnorm = {
      1:  1.192092800768788e-07,
      2:  5.978858893805233e-04,
      4:  5.116619363445086e-02,
      8:  5.800524627688768e-01,
      12: 1.461661507209034e+00,
      18: 3.010066362817634e+00
    }
  else:
    pcm.utils.raise_err(
      ValueError,
      f"'linalg.expm' does not support matrices of type '{dtype}'."
    )
  if (deg is not None):
    return maxnorm[deg]
  else:
    return maxnorm

def _compute_l1_norm(A):
  l1_norm = tf.math.reduce_sum(tf.math.abs(A), axis=-2, keepdims=True)
  return tf.math.reduce_max(l1_norm, axis=-1, keepdims=True)

# Taylor approximations
# =====================================
def _taylor_px(I, A, deg, l1_norm, max_nb_squarings):
  if (deg == 1):
    return _taylor_p1(I, A)
  elif (deg == 2):
    return _taylor_p2(I, A)
  elif (deg == 4):
    return _taylor_p4(I, A)
  elif (deg == 8):
    return _taylor_p8(I, A)
  elif (deg == 12):
    return _taylor_p12(I, A)
  else:
    maxnorm = _get_maxnorm(A.dtype, deg=18)
    squarings, As, is_finite = _scaling(A, l1_norm, maxnorm, max_nb_squarings)
    eA = _taylor_p18(I, As)
    eA = _squaring(eA, squarings, is_finite)
    return eA

# Linear combination
# -------------------------------------
def _linear_comb(scalars, *tensors):
  y = tf.zeros_like(tensors[0])
  scalars = tf.split(scalars, scalars.shape[0])
  scalars = tf.nest.map_structure(tf.squeeze, scalars)
  for (i, s) in enumerate(scalars):
    if tf.math.not_equal(x=s, y=tf.constant(0, dtype=s.dtype)):
      if tf.math.equal(x=s, y=tf.constant(1, dtype=s.dtype)):
        y += tensors[i]
      else:
        y += s * tensors[i]
  return y

# Polynomial of degree p=1
# -------------------------------------
def _taylor_p1(I, A):
  return I + A

# Polynomial of degree p=2
# -------------------------------------
def _taylor_p2(I, A):
  A2 = tf.matmul(A, A)
  return I + A + 0.5 * A2

# Polynomial of degree p=4
# -------------------------------------
def _taylor_p4_coeff():
  # Initialize coefficients
  a = np.ones(3, dtype=np.float64)
  a[0] = 1/2
  a[1] = 1/6
  a[2] = 1/24
  # Return coefficients
  return {
    "a": tf.constant(a)
  }
_c4 = tf.nest.map_structure(tf.squeeze, _taylor_p4_coeff())

def _taylor_p4(I, A):
  c = tf.nest.map_structure(lambda x: tf.cast(x, dtype=A.dtype), _c4)
  A2 = tf.matmul(A, A)
  A4 = tf.matmul(A2, _linear_comb(c["a"], I, A, A2))
  return I + A + A4

# Polynomial of degree p=8
# -------------------------------------
def _taylor_p8_coeff():
  # Define constant
  f = np.sqrt(177)
  # Initialize coefficients
  # > 'x' coefficients
  x = np.ones(2, dtype=np.float64)
  x[0] = 2/3
  # > 'a' coefficients
  a = np.ones(2, dtype=np.float64)
  a[0] = x[0] * ((1 + f) / 88)
  a[1] = x[0] * ((1 + f) / 352)
  # > 'b' coefficients
  b = np.ones(4, dtype=np.float64)
  b[0] = (-271 + 29 * f) / (315 * x[0])
  b[1] = (-11 + 11 * f) / (1260 * x[0])
  b[2] = (-99 + 11 * f) / (5040 * x[0])
  b[3] = (89 - f) / (5040 * x[0]**2)
  # > 'c' coefficients
  c = np.ones(4, dtype=np.float64)
  c[2] = (857 - 58 * f) / 630
  # Return coefficients
  return {
    "x": tf.constant(x),
    "a": tf.constant(a),
    "b": tf.constant(b),
    "c": tf.constant(c)
  }
_c8 = tf.nest.map_structure(tf.squeeze, _taylor_p8_coeff())

def _taylor_p8(I, A):
  c = tf.nest.map_structure(lambda x: tf.cast(x, dtype=A.dtype), _c8)
  A2 = tf.matmul(A, A)
  A4 = tf.matmul(A2, _linear_comb(c["a"], A, A2))
  A8 = tf.matmul(
    _linear_comb(c["x"], A2, A4),
    _linear_comb(c["b"], I, A, A2, A4)
  )
  return _linear_comb(c["c"], I, A, A2, A8)

# Polynomial of degree p=12
# -------------------------------------
def _taylor_p12_coeff():
  # Initialize coefficients
  a = np.zeros((4,4), dtype=np.float64)
  # > First row
  a[0,0] = - 0.01860232051462055322434373004330
  a[0,1] = - 0.00500702322573317730979741843919
  a[0,2] = - 0.57342012296052226390595242078900
  a[0,3] = - 0.13339969394389205970076892698300
  # > Second row
  a[1,0] =   4.60000000000000000000000000000000
  a[1,1] =   0.99287510353848683614047957150500
  a[1,2] = - 0.13244556105279963884507499745400
  a[1,3] =   0.00172990000000000000000000000000
  # > Third row
  a[2,0] =   0.21169311829980944294932332333600
  a[2,1] =   0.15822438471572672537176889325200
  a[2,2] =   0.16563516943672741501117166841900
  a[2,3] =   0.01078627793157924250263206401080
  # > Fourth row
  a[3,1] = - 0.13181061013830184015681934946400
  a[3,2] = - 0.02027855540589259079335682299450
  a[3,3] = - 0.00675951846863086359778560766482
  # Return coefficients
  return {
    "a": tf.split(tf.constant(a), a.shape[0], axis=0)
  }
_c12 = tf.nest.map_structure(tf.squeeze, _taylor_p12_coeff())

def _taylor_p12(I, A):
  # Prepare coefficients
  c = tf.nest.map_structure(lambda x: tf.cast(x, dtype=A.dtype), _c12)
  # Compute
  A2 = tf.matmul(A, A)
  A3 = tf.matmul(A2, A)
  Bs = [_linear_comb(ai, I, A, A2, A3) for ai in c["a"]]
  A6 = Bs[2] + tf.matmul(Bs[3], Bs[3])
  return Bs[0] + tf.matmul(Bs[1] + A6, A6)

# Polynomial of degree p=18
# -------------------------------------
def _taylor_p18_coeff():
  # Initialize coefficients
  a = np.zeros((5,5), dtype=np.float64)
  # > First row
  a[0,1] = - 0.1003655810301446200146149394050000
  a[0,2] = - 0.0080292464824115696011691951524400
  a[0,3] = - 0.0008921384980457299556854661280490
  # > Second row
  a[1,1] =   0.39784974949964507614519612771028457569650810840768562238459516076401453731490320304046603397034261700000
  a[1,2] =   1.36783778460411719922523706878222824210645354065491579526746233270700047628463874573881208276186045800000
  a[1,3] =   0.49828962252538267755685881726227335618491813973196962699234507731797445789066754277076183775043055610000
  a[1,4] = - 0.00063789819459472330924155005649192855187738275810133320551486532413531207896463231869653983175231947600
  # > Third row
  a[2,0] = -10.96763960529620625935174626753684124863041876254774214673058385106461743913502064396554589372626845000000
  a[2,1] =   1.68015813878906197182785401724810509527857959301056685809158587562736474789572407003358680294743615700000
  a[2,2] =   0.05717798464788655127028717132252481317274975093707600394506105236273081373356534970690710643085727120000
  a[2,3] = - 0.00698210122488052084290466483801553166438983815207775325835344712560520533528896627725791892588133783400
  a[2,4] =   0.00003349750170860705383133673406684398020225996444991565389728295589367037178816169580298011831485225359
  # > Fourth row
  a[3,0] = - 0.09043168323908105619714688770183495499470866281162649735086602288456671216199491949073419844120202066000
  a[3,1] = - 0.06764045190713819075600799003797074854562826013273060070581796689746210078264908587143648477465431597000
  a[3,2] =   0.06759613017704596460827991950789370565071000768138331455747136830953224978586305376886338214283464385000
  a[3,3] =   0.02955525704293155274260691822422312437473046472743510159951445245685893720550532525991666174704105350000
  a[3,4] = - 0.00001391802575160607011247399793437424028074102305234870853220069004278819595344896368546425681168813708
  # > Fifth row
  a[4,2] = - 0.0923364619367118592764570775143000
  a[4,3] = - 0.0169364939002081717191385115723000
  a[4,4] = - 0.0000140086798182036159794363205726
  # Return coefficients
  return {
    "a": tf.split(tf.constant(a), a.shape[0], axis=0)
  }
_c18 = tf.nest.map_structure(tf.squeeze, _taylor_p18_coeff())

def _taylor_p18(I, A):
  c = tf.nest.map_structure(lambda x: tf.cast(x, dtype=A.dtype), _c18)
  A2 = tf.matmul(A, A)
  A3 = tf.matmul(A2, A)
  A6 = tf.matmul(A3, A3)
  Bs = [_linear_comb(ai, I, A, A2, A3, A6) for ai in c["a"]]
  A9 = tf.matmul(Bs[0], Bs[4]) + Bs[3]
  return Bs[1] + tf.matmul(Bs[2] + A9, A9)

# Scaling and squaring operations
# -------------------------------------
def _scaling(A, l1_norm, maxnorm, max_nb_squarings):
  is_finite = tf.math.is_finite(tf.math.reduce_max(l1_norm))
  if is_finite:
    two = tf.constant(2.0, l1_norm.dtype)
    # Compute scaling
    ov_log2 = 1/tf.math.log(two)
    squarings = ov_log2 * tf.math.log((1/maxnorm) * l1_norm)
    squarings = tf.math.maximum(tf.math.floor(squarings), 0)
    # Apply scaling
    As = (1 / tf.math.pow(two, squarings)) * A
  else:
    squarings, As = tf.zeros_like(l1_norm), A
  if ((max_nb_squarings is not None) and (max_nb_squarings >= 0)):
    squarings = tf.math.minimum(squarings, max_nb_squarings)
  return squarings, As, is_finite

def _squaring(eA, squarings, is_finite):
  # Maximum number of squarings operation
  max_squarings = tf.math.reduce_max(squarings)
  if (is_finite and (max_squarings > 0)):
    # Define stop condition: max squaring reached
    def cond(i, _):
      return tf.math.less(i, max_squarings)
    # Define squaring operation
    def body(i, e):
      j = i + 1
      e = tf.where(
        tf.math.less(i, squarings),
        x=tf.matmul(e, e),
        y=e
      )
      return j, e
    # Perform squarings
    i = tf.constant(0.0, max_squarings.dtype)
    _, eA = tf.while_loop(
      cond=cond,
      body=body,
      loop_vars=[i, eA]
    )
  return eA
