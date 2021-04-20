import tensorflow.keras.regularizers as reg

def get(name, l1, l2):
    return (
        reg.L1(l1=l1)
        if name == "l1"
        else reg.L2(l2=l2)
        if name == "l2"
        else reg.L1L2(l1=l1, l2=l2)
        if name == "l1+l2"
        else None
    )
