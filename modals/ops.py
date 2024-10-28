import numpy as np
import torch
from itertools import combinations


def cosine(X, mu):
    Xm = torch.matmul(X, mu)
    norm_p = torch.norm(X, dim=1) * torch.norm(mu)
    distances = 1 - Xm / (norm_p + 1e-5)
    return distances


def pdist(vectors):
    squared_norms = vectors.pow(2).sum(dim=1)
    distance_matrix = squared_norms.unsqueeze(0) + squared_norms.unsqueeze(1) - 2 * vectors.mm(vectors.t())
    return distance_matrix


class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllPositivePairSelector:
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples.
    """

    def __init__(self, balance=True):
        super(AllPositivePairSelector, self).__init__()
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        labels = labels.to(embeddings.device)  # Ensure labels are on the same device
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs).to(embeddings.device)  # Move pairs to the appropriate device

        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero(as_tuple=True)]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero(as_tuple=True)]

        if self.balance:
            # Randomly sample negative pairs to match the number of positive pairs
            sampled_negatives = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]
            return positive_pairs, sampled_negatives

        return positive_pairs, negative_pairs


class HardNegativePairSelector:
    """
    Creates all possible positive pairs. For negative pairs, pairs with the smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self):
        super(HardNegativePairSelector, self).__init__()

    def get_pairs(self, embeddings, labels):
        # Calculate distance matrix using the provided pdist function
        distance_matrix = pdist(embeddings)

        labels = labels.to(embeddings.device)  # Ensure labels are on the same device

        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs).to(embeddings.device)  # Move pairs to the appropriate device

        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero(as_tuple=True)]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero(as_tuple=True)]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        top_negatives_indices = torch.topk(negative_distances, len(positive_pairs), largest=False).indices
        top_negative_pairs = negative_pairs[top_negatives_indices]

        return positive_pairs, top_negative_pairs


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector:
    """
    Returns all possible triplets.
    May be impractical in most cases.
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.to(embeddings.device)  # Ensure labels are on the same device
        triplets = []

        # Get unique labels and iterate over them
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            label_mask = (labels == label)
            label_indices = torch.nonzero(label_mask).flatten()  # Get indices of positive samples

            if len(label_indices) < 2:
                continue

            negative_indices = torch.nonzero(~label_mask).flatten()  # Get indices of negative samples

            # Generate all anchor-positive pairs
            anchor_positives = combinations(label_indices.cpu().numpy(), 2)  # Convert to numpy for combinations

            # Add all negatives for all positive pairs
            for anchor_positive in anchor_positives:
                for neg_ind in negative_indices:
                    triplets.append([anchor_positive[0].item(), anchor_positive[1].item(), neg_ind.item()])

        return torch.LongTensor(triplets).to(embeddings.device)


def hardest_negative(loss_values):
    hard_negative = torch.argmax(loss_values)
    return hard_negative.item() if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = torch.where(loss_values > 0)[0]
    return hard_negatives[torch.randint(0, len(hard_negatives), (1,))].item() if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = torch.where((loss_values < margin) & (loss_values > 0))[0]
    return semihard_negatives[torch.randint(0, len(semihard_negatives), (1,))].item() if len(
        semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector:
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet.
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair.
    """

    def __init__(self, margin, negative_selection_fn):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        distance_matrix = pdist(embeddings)

        labels = labels.to(embeddings.device)  # Ensure labels are on the same device
        triplets = []

        if len(torch.unique(labels)) == 1:
            return torch.LongTensor(triplets).to(embeddings.device)

        for label in torch.unique(labels):
            label_mask = (labels == label)
            label_indices = torch.nonzero(label_mask).flatten()  # Get indices of positive samples

            if len(label_indices) < 2:
                continue

            negative_indices = torch.nonzero(~label_mask).flatten()  # Get indices of negative samples

            # All anchor-positive pairs
            anchor_positives = combinations(label_indices.cpu().numpy(), 2)
            anchor_positives = np.array(list(anchor_positives))

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[anchor_positive[0], negative_indices] + self.margin

                loss_values = loss_values.cpu().numpy()  # Convert to numpy for the selection function
                hard_negative_idx = self.negative_selection_fn(loss_values)

                if hard_negative_idx is not None:
                    hard_negative = negative_indices[hard_negative_idx]
                    triplets.append([anchor_positive[0].item(), anchor_positive[1].item(), hard_negative.item()])

        if len(triplets) == 0:
            # Fallback if no triplets are found
            if len(anchor_positive) < 2 and len(negative_indices) < 1:
                triplets.append([anchor_positive, anchor_positive, negative_indices])
            elif len(anchor_positive) >= 2 and len(negative_indices) < 1:
                triplets.append([anchor_positive[0], anchor_positive[1], negative_indices])
            else:
                triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        return torch.LongTensor(triplets).to(embeddings.device)  # Return as tensor on the correct device


def HardestNegativeTripletSelector(margin):
    return FunctionNegativeTripletSelector(margin=margin, negative_selection_fn=hardest_negative)


def RandomNegativeTripletSelector(margin):
    return FunctionNegativeTripletSelector(margin=margin, negative_selection_fn=random_hard_negative)


def SemihardNegativeTripletSelector(margin):
    return FunctionNegativeTripletSelector(margin=margin, negative_selection_fn=lambda x: semihard_negative(x, margin),
                                           )
