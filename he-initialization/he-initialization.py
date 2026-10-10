import math

def he_initialization(W: list, fan_in: int) -> list:
    """
    Returns the weights mapped to the He uniform range.
    """
    L = math.sqrt(6 / fan_in)
    scaled_W = [[round(val*2*L-L, 4) for val in row] for row in W]
    return scaled_W
    pass