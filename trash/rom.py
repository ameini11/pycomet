import numpy as np


class ROM(object):

  def __init__(
    self,
    phi,
    xref,
    xscale,
    use_log=True
  ):
    self.phi = phi
    self.xref = xref
    self.xscale = xscale
    self.use_log = use_log

  def autoencode(self, x):
    return self.decode(self.encode(x))

  def encode(self, x):
    # Transform
    if self.use_log:
      x = np.log(x)
    x = (x - self.xref) / self.xscale
    # Encode
    return x @ self.phi

  def decode(self, z):
    # Decode
    x = z @ self.phi.T
    # Transform
    x = x * self.xscale + self.xref
    if self.use_log:
      x = np.exp(x)
    return x

  def save(self, path):
    for (k, v) in (
      ("phi", self.phi),
      ("xref", self.xref.reshape(-1,1)),
      ("xscale", self.xscale.reshape(-1,1))
    ):
      np.savetxt(path + f"/{k}.txt", v, fmt='%.18e')
