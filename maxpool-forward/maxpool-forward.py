def maxpool_forward(X: list, pool_size: int, stride: int) -> list:
    """
    Returns the maximum value from every pooling window.
    """
    H = len(X)
    W = len(X[0])
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    output = []
    for i in range(H_out):
        row = []
        for j in range(W_out):
            start_r = i * stride
            start_c = j * stride
            max_val = X[start_r][start_c]
            for a in range(pool_size):
                for b in range(pool_size):
                    val = X[start_r + a][start_c + b]
                    if val>max_val:
                        max_val = val
            row.append(max_val)
        output.append(row)
    return output
    pass