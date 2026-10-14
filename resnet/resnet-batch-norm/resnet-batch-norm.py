import numpy as np

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns the normalized residual-block result and selected mode in a dictionary.
    """
    x = np.array(x, dtype=np.float64)
    W1 = np.array(W1, dtype=np.float64)
    W2 = np.array(W2, dtype=np.float64)
    gamma1 = np.array(gamma1, dtype=np.float64)
    beta1 = np.array(beta1, dtype=np.float64)
    gamma2 = np.array(gamma2, dtype=np.float64)
    beta2 = np.array(beta2, dtype=np.float64)
    
    eps = 1e-5

    def batch_norm(arr, gamma, beta):
        mean = arr.mean(axis=0)
        var = arr.var(axis=0)
        x_hat = (arr - mean) / np.sqrt(var + eps)
        return gamma * x_hat + beta

    def relu(arr):
        return np.maximum(0, arr)
    if mode == "post":
        residual = x
        out = np.dot(x, W1)
        out = batch_norm(out, gamma1, beta1)
        out =relu(out)
        out = np.dot(out, W2)
        out = batch_norm(out, gamma2, beta2)
        out = out+ residual
        out = relu(out)
    elif mode == "pre":
        residual = x
        out = batch_norm(x, gamma1, beta1)
        out = relu(out)
        out = np.dot(out, W1)
        out = batch_norm(out, gamma2, beta2)
        out = relu(out)
        out = np.dot(out, W2)
        out = out + residual
    output_rounded = np.round(out,4).tolist()
    return{
        "output": output_rounded,
        "mode": mode
    }
    pass