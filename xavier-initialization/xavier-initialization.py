import math

def xavier_initialization(W: list, fan_in: int, fan_out: int) -> list:
    """
    Returns the weights mapped to the Xavier uniform range.
    """
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    scaled_W = []
    for row in W:
        scaled_row = [val*(2*limit) - limit for val in row]
        scaled_W.append(scaled_row)
    return scaled_W
    pass