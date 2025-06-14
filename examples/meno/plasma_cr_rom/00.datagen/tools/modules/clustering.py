import copy
import numpy as np

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Clusterer(object):

  def __init__(
    self,
    X,
    xform=None,
    scaling="minmax",
    clustering="km",
    n_clusters=2,
    random_state=0
  ):
    # Data
    self.X = np.array(X)
    # Transformation
    self.xform = xform
    # Scaling
    self.scaler = None
    if (scaling is not None):
      self.scaler = self.construct_scaler(method=scaling)
    # Clustering
    self.clustering = clustering
    self.clusterer = self.construct_clusterer(
      X=self.X,
      method=clustering,
      n_clusters=n_clusters,
      random_state=random_state
    )

  @property
  def centers(self):
    if (self.clustering == 'gm'):
      self._centers = self.clusterer.means_
    else:
      self._centers = self.clusterer.cluster_centers_
    self._centers = self.scaler.inverse_transform(self._centers)
    return self._centers

  # Scaling
  # -----------------------------------
  def construct_scaler(
    self,
    method
  ):
    X = copy.deepcopy(self.X)
    if (self.xform is not None):
      X = self.xform["dir"](X)
    if (method == "std"):
      scaler = StandardScaler().fit(X)
    else:
      scaler = MinMaxScaler().fit(X)
    del X
    return scaler

  def transform(
    self,
    X
  ):
    if (self.xform is not None):
      X = self.xform(X)
    if (self.scaler is not None):
      X = self.scaler.transform(X)
    return X

  # Clustering
  # -----------------------------------
  def construct_clusterer(
    self,
    X,
    method,
    n_clusters,
    random_state=0
  ):
    X = self.transform(X)
    if (method == "gm"):
      clusterer = GaussianMixture(
        n_components=n_clusters,
        random_state=random_state,
        init_params="k-means++"
      ).fit(X)
    else:
      clusterer = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        init="k-means++"
      ).fit(X)
    return clusterer

  def cluster(
    self,
    X
  ):
    X = self.transform(X)
    return self.clusterer.predict(X)
