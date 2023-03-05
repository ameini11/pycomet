import itertools

from pycomet import utils
from tensorflow.python.training.tracking.data_structures import NoDependency


class GroupsConnect(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        groups_graph
        *args,
        **kwargs
    ):
        self.groups_graph = groups_graph
        self.fastest_to_current_ts = self.connect_fastest_to_current_ts(
            groups_graph
        )
        self.current_to_next_ts = NoDependency(
            self.connect_current_to_next_ts(groups_graph)
        )

    # Groups connectivity/indexing
    ###########################################################################
    def connect_current_to_next_ts(self, graph, level=0, connectivity={}):
        d = {}
        connectivity[level] = []
        if isinstance(graph, dict):
            for g in graph.values():
                if hasattr(g, "keys"):
                    l = list(g.keys())
                    d = {**d, **g}
                else:
                    l = d = g
                connectivity[level].append(l)
        else:
            connectivity[level] = None
        if d != {}:
            return self.connect_current_to_next_ts(
                d, level=level+1, connectivity=connectivity
            )
        else:
            return connectivity

    def connect_fastest_to_current_ts(self, graph, ts={}):
        nb_ts = utils.depth(graph)
        connectivity = self.graph_to_nested_list(graph)
        for l in range(-1, nb_ts):
            ts[l+1] = self.merge(connectivity, it_max=l+1)
        return ts

    def graph_to_nested_list(self, graph):
        if isinstance(graph, dict):
            ls = []
            for k, v in graph.items():
                ls.append(self.graph_to_nested_list(v))
            return ls
        else:
            if isinstance(graph, (list, tuple, set)):
                return list(graph)
            else:
                utils.raise_value_err(
                    "The last level of the groups graph has to be a list."
                )

    def merge(self, ls, it=0, it_max=1):
        if any(isinstance(l, list) for l in ls):
            if it < it_max:
                ls = list(itertools.chain.from_iterable(ls))
                return self.merge(ls, it=it+1, it_max=it_max)
            else:
                return self.join(ls)

    def join(self, ls):
        try:
            ls = [list(itertools.chain.from_iterable(l)) for l in ls]
            return self.join(ls)
        except:
            return ls

    
# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------
#
# groups_graph = {
#     0: {
#         0: [0,1,2],
#         1: [3,4,5],
#         2: [6,7,8]
#     }
# }
#
# fastest_to_current_ts = {
#     0: [
#         [0, 1, 2, 3, 4, 5, 6, 7, 8]
#     ],
#     1: [
#         [0, 1, 2],
#         [3, 4, 5],
#         [6, 7, 8]
#     ],
#     2: None
# }
#
# current_to_next_ts = {
#     0: [
#         [0, 1, 2]
#     ],
#     1: [
#         [0, 1, 2],
#         [3, 4, 5],
#         [6, 7, 8]
#     ],
#     2: None
# }
#
# -----------------------------------------------------------------------------