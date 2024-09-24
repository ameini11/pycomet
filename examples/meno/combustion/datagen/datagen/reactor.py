import json
import numpy as np
import scipy as sp


class POLLUReactor(object):

  def __init__(
    self,
    species_file,
    reactions_file,
  ) -> None:
    with open(species_file) as file:
      self.species_names = json.load(file)
    self.species_to_index = self.get_species_to_index()
    self.species_indices = self.get_species_indices()
    self.nb_eqs = len(self.species_names["full"])
    with open(reactions_file) as file:
      self.reactions = json.load(file)["reactions"]
    self.nb_reactions = len(self.reactions)
    self._set_rates()
    # Matrix exponential integrator with linear/nonlinear splitting
    self.amat_indices = self._set_amat_indices()

  def get_species_to_index(self):
    return {name: i for (i, name) in enumerate(self.species_names["full"])}

  def get_species_indices(self):
    species_indices = {}
    for (k, species) in self.species_names.items():
      species_indices[k] = np.sort([self.species_to_index[s] for s in species])
    species_indices["lin"] = np.concatenate(
      [species_indices[f"lin_o{o}"] for o in range(2)]
    )
    species_indices["lin"] = np.sort(np.unique(species_indices["lin"]))
    return species_indices

  # Reactions
  # -----------------------------------
  def _set_rates(self):
    # Rates
    k = np.zeros(self.nb_reactions)
    # Minimum index
    r_min = np.array(list(self.reactions.keys())).astype(np.int32).min()
    # Species indices and stoichiometric coefficients
    ireac, iprod, creac, cprod = [], [], [], []
    for (r, reaction) in self.reactions.items():
      # > Rate
      k[int(r)-r_min] = reaction["k"]
      # > Reactants
      rreac = self._set_side(reaction["reac"])
      ireac.append(rreac[0])
      creac.append(rreac[1])
      # > Products
      rprod = self._set_side(reaction["prod"])
      iprod.append(rprod[0])
      cprod.append(rprod[1])
    self.rates = {
      "k": k,
      "reac": {
        "i": np.vstack(ireac).astype(bool),
        "c": np.vstack(creac)
      },
      "prod": {
        "i": np.vstack(iprod).astype(bool),
        "c": np.vstack(cprod)
      }
    }

  def _set_side(self, side):
    # Species indices and stoichiometric coefficients
    indices, coeff = [np.zeros(self.nb_eqs) for _ in range(2)]
    for (s, c) in side.items():
      indices[self.species_to_index[s]] = 1.0
      coeff[self.species_to_index[s]] = float(c)
    return indices, coeff

  # ODE Solver
  # -----------------------------------
  def solve(self, t, y0):
    return sp.integrate.solve_ivp(
      fun=self.rhs,
      t_span=[0.0,t[-1]],
      y0=y0,
      method="LSODA",
      t_eval=t,
      first_step=1e-15,
      rtol=1e-12,
      atol=1e-20
    ).y.T

  def rhs(self, t, y):
    f = np.zeros_like(y)
    for r in range(self.nb_reactions):
      # Species indices and stoichiometric coefficients
      # > Reactants
      ir, cr = [self.rates["reac"][k][r] for k in ("i", "c")]
      # > Products
      ip, cp = [self.rates["prod"][k][r] for k in ("i", "c")]
      # Source term
      w = self.rates["k"][r]*np.prod(y[ir])
      # Update RHS
      np.add.at(f, ir, -cr[ir]*w)
      np.add.at(f, ip,  cp[ip]*w)
    return f

  # Matrix exponential integrator with linear/nonlinear splitting
  # -----------------------------------
  def _set_amat_indices(self):
    s_l = self.species_indices["lin"]
    nb_eqs = len(s_l)
    ireac, iprod, creac, cprod = [], [], [], []
    for r in range(self.nb_reactions):
      # Species indices and stoichiometric coefficients
      # > Reactants
      ir_1d, cr_1d = [self.rates["reac"][k][r] for k in ("i", "c")]
      # > Products
      ip_1d, cp_1d = [self.rates["prod"][k][r] for k in ("i", "c")]
      if (np.sum(ir_1d[s_l]) == 0):
        ir = cr = ip = cp = np.zeros(nb_eqs**2)
      else:
        ir = np.outer(ir_1d[s_l],ir_1d[s_l]).reshape(-1)
        ip = np.outer(ip_1d[s_l],ir_1d[s_l]).reshape(-1)
        cr = np.outer(cr_1d[s_l],ir_1d[s_l]).reshape(-1)
        cp = np.outer(cp_1d[s_l],ir_1d[s_l]).reshape(-1)
      ireac.append(ir)
      iprod.append(ip)
      creac.append(cr)
      cprod.append(cp)
    return {
      "reac": {
        "i": np.vstack(ireac).astype(bool),
        "c": np.vstack(creac)
      },
      "prod": {
        "i": np.vstack(iprod).astype(bool),
        "c": np.vstack(cprod)
      }
    }

  def compose_lin_ops(self, y_nl):
    s_l, s_nl = [self.species_indices[k] for k in ("lin", "nlin")]
    nb_eqs = len(s_l)
    a, b = np.zeros(nb_eqs**2), np.zeros(nb_eqs)
    for r in range(self.nb_reactions):
      # Source term
      ir = self.rates["reac"]["i"][r]
      w = self.rates["k"][r]*np.prod(y_nl[ir[s_nl]])
      # Update linear operators
      if (np.sum(ir[s_l]) == 0):
        b = self._update_bvec(b, r, w, s_l)
      else:
        a = self._update_amat(a, r, w)
    return a.reshape(nb_eqs,nb_eqs), b

  def _update_bvec(self, b, r, w, s_l):
    # Species indices and stoichiometric coefficients
    # > Reactants
    ir, cr = [self.rates["reac"][k][r] for k in ("i", "c")]
    # > Products
    ip, cp = [self.rates["prod"][k][r] for k in ("i", "c")]
    # Update 'b' vector
    np.add.at(b, ir[s_l], -cr[s_l][ir[s_l]]*w)
    np.add.at(b, ip[s_l],  cp[s_l][ip[s_l]]*w)
    return b

  def _update_amat(self, a, r, w):
    # Species indices and stoichiometric coefficients
    # > Reactants
    ir, cr = [self.amat_indices["reac"][k][r] for k in ("i", "c")]
    # > Products
    ip, cp = [self.amat_indices["prod"][k][r] for k in ("i", "c")]
    # Update 'a' vector
    np.add.at(a, ir, -cr[ir]*w)
    np.add.at(a, ip,  cp[ip]*w)
    return a
