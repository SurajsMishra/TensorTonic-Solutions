def linear_layer_forward(X: list, W: list, b: list) -> list:
    """
    Returns the affine transformation for every input row.
    """
    n = len(X)
    d_in = len(X[0])
    d_out = len(W[0])
    Y = [[0.0 for i in range(d_out)] for i in range(n)]
    for i in range(n):
        for j in range(d_out):
            dot_product = sum(X[i][k]*W[k][j] for k in range(d_in))
            Y[i][j] = dot_product + b[j]
    return Y
    pass