import numpy as np
from sklearn.metrics import make_scorer
from sklearn.naive_bayes import LabelBinarizer

def calculate_dcg(scores):
    """Calculate DCG based on the list of relevance scores."""
    return np.sum(
        [(2**score - 1) / np.log2(idx + 2) for idx, score in enumerate(scores)]
    )

def calculate_ndcg(pred_probs, true_labels, k=5):
    """Calculate the nDCG score for one instance."""
    # Get indices of the top k predictions
    top_k_indices = np.argsort(pred_probs)[::-1][:k]
    # Create relevance scores: 1 if correct, 0 otherwise
    relevance_scores = [1 if idx == true_labels else 0 for idx in top_k_indices]
    # Calculate DCG
    dcg = calculate_dcg(relevance_scores)
    # Calculate IDCG: Ideal scenario where the true label is at the top
    idcg = calculate_dcg([1] + [0]*(k-1))
    # Return nDCG
    return dcg / idcg if idcg > 0 else 0

def ndcg(all_pred_probs, all_true_indices, k=5):
    """Calculate the average nDCG score for multiple predictions."""
    ndcg_scores = [calculate_ndcg(pred_probs, true_idx, k) 
                   for pred_probs, true_idx in zip(all_pred_probs, all_true_indices)]
    return np.mean(ndcg_scores)


# NDCG Score Metric Function
# Reference: https://www.kaggle.com/davidgasquez/ndcg-scorer

def dcg_score(y_true, y_score, k=5):
    
    """Discounted cumulative gain (DCG) at rank K.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array, shape = [n_samples, n_classes]
        Predicted scores.
    k : int
        Rank.

    Returns
    -------
    score : float
    """
    
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)


def ndcg_score_udf(ground_truth, predictions, k=5):
    
    """Normalized discounted cumulative gain (NDCG) at rank K.

    Normalized Discounted Cumulative Gain (NDCG) measures the performance of a
    recommendation system based on the graded relevance of the recommended
    entities. It varies from 0.0 to 1.0, with 1.0 representing the ideal
    ranking of the entities.

    Parameters
    ----------
    ground_truth : array, shape = [n_samples]
        Ground truth (true labels represended as integers).
    predictions : array, shape = [n_samples, n_classes]
        Predicted probabilities.
    k : int
        Rank.

    Returns
    -------
    score : float

    Example
    -------
    >>> ground_truth = [1, 0, 2]
    >>> predictions = [[0.15, 0.55, 0.2], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
    >>> score = ndcg_score(ground_truth, predictions, k=2)
    1.0
    >>> predictions = [[0.9, 0.5, 0.8], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
    >>> score = ndcg_score(ground_truth, predictions, k=2)
    0.6666666666
    """
    
    lb = LabelBinarizer()
    lb.fit(range(predictions.shape[1] + 1))
    T = lb.transform(ground_truth)

    scores = []

    # Iterate over each y_true and compute the DCG score
    for y_true, y_score in zip(T, predictions):
        actual = dcg_score(y_true, y_score, k)
        best = dcg_score(y_true, y_true, k)
        score = float(actual) / float(best)
        scores.append(score)

    return np.mean(scores)

# Create the scorer with NDCG score metric function
ndcg_scorer = make_scorer(ndcg_score_udf, needs_proba=True, k=5)