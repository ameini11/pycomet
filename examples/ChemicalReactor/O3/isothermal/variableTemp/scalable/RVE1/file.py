
import itertools



graph = {
    1: {
        1: {
            1: [1,2,3],
            2: [4,5,6],
            3: [7,8,9]
        },
        2: {
            4: [10,11,12],
            5: [13,14,15],
            6: [16,17,18]
        },
        3: {
            7: [19,20,21],
            8: [22,23,24],
            9: [25,26,27]
        }
    },
    2: {
        1: {
            1: [1,2,3],
            2: [4,5,6],
            3: [7,8,9]
        },
        2: {
            4: [10,11,12],
            5: [13,14,15],
            6: [16,17,18]
        },
        3: {
            7: [19,20,21],
            8: [22,23,24],
            9: [25,26,27]
        }
    }
}





def groups_connectivity(graph):
    if isinstance(graph, dict):
        conn = []
        for k, v in graph.items():
            conn.append(groups_connectivity(v))
        return conn
    else:
        if isinstance(graph, (list, tuple, set)):
            return list(graph)
        else:
            raise ValueError(
                "The last level of the groups connectivity has to be a list."
            )




def groups_time_scales(connectivity, levels, ts={}):
    for l in range(levels):
        ts[l] = merge(connectivity, it_max=l)
    return ts




def join(ls):
    try:
        ls = [list(itertools.chain.from_iterable(l)) for l in ls]
        return join(ls)
    except:
        return ls


def merge(ls, it=0, it_max=1):
    if any(isinstance(l, list) for l in ls):
        if it < it_max:
            ls = list(itertools.chain.from_iterable(ls))
            return merge(ls, it=it+1, it_max=it_max)
        else:
            return join(ls)


print(groups_connectivity(graph))


print(groups_time_scales(groups_connectivity(graph), 3))