from pycomet import utils


def unpack_graph(
  graph,
  edge_index=None
):
  if (edge_index is not None):
    node_attr, edge_attr = utils.unpack_list(graph, nb_elem=2)
  else:
    node_attr, edge_index, edge_attr = utils.unpack_list(graph, nb_elem=3)
  return node_attr, edge_index, edge_attr
