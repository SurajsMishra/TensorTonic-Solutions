import math

def gaussian_naive_bayes(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns a predicted class label for every test sample.
    """
    n_samples = len(y_train)
    n_features = len(X_train[0])
    eps = 1e-9

    # 1. Unique classes
    classes = list(set(y_train))
    
    # 2. Group data by class
    class_data = {c: [] for c in classes}
    for row, label in zip(X_train, y_train):
        class_data[label].append(row)

    priors = {}
    means = {}
    variances = {}

    # 3. Compute statistics per class
    for c in classes:
        rows = class_data[c]
        n_c = len(rows)
        
        priors[c] = n_c / n_samples
        
        # Mean per feature
        means[c] = [
            sum(rows[i][j] for i in range(n_c)) / n_c 
            for j in range(n_features)
        ]
        
        # Population variance per feature + eps
        variances[c] = [
            (sum((rows[i][j] - means[c][j]) ** 2 for i in range(n_c)) / n_c) + eps
            for j in range(n_features)
        ]

    # 4. Predict test set
    predictions = []
    for sample in X_test:
        best_class = None
        max_log_posterior = -float('inf')

        for c in classes:
            log_posterior = math.log(priors[c])
            for j in range(n_features):
                x_j = sample[j]
                mu = means[c][j]
                var = variances[c][j]
                
                likelihood = -0.5 * math.log(2 * math.pi * var) - ((x_j - mu) ** 2) / (2 * var)
                log_posterior += likelihood

            if log_posterior > max_log_posterior:
                max_log_posterior = log_posterior
                best_class = c

        predictions.append(best_class)

    return predictions