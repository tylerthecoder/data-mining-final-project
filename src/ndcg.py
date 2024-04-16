import numpy as np

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
