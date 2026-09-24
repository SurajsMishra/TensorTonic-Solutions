import numpy as np

def wasserstein_critic_loss(real_scores: list, fake_scores: list) -> float:
    """
    Returns the loss as a float.
    """
    mean_fake = np.mean(fake_scores)
    mean_real = np.mean(real_scores)
    return float(mean_fake - mean_real)
    pass