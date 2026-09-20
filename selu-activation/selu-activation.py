import math

def selu(x: list) -> list:
    """
    Returns SELU values rounded to four decimal places.
    """
    # Write code here
    scale = 1.0507
    alpha = 1.6733
    result = []
    for val in x:
        if val > 0:
            res = scale* val
        else:
            res = scale * alpha * (math.exp(val) - 1.0)
        result.append(round(res, 4))

    return result
    pass