import tensorflow.keras.regularizers as reg

def get(identifier):
    if identifier is None:
        return None
    name, scales = identifier[0], identifier[1:]
    return (
        reg.l1(l=scales[0])
        if name == "l1"
        else reg.l2(l=scales[0])
        if name == "l2"
        else reg.l1_l2(l1=scales[0], l2=scales[1])
        if name == "l1+l2"
        else None
    )
