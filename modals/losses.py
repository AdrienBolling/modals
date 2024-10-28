import torch
import torch.nn as nn
import torch.nn.functional as F


def discriminator_loss(d_real: torch.Tensor, d_fake: torch.Tensor, eps: float) -> torch.Tensor:
    """Calculate discriminator loss."""
    return -torch.mean(torch.log(d_real + eps) + torch.log(1 - d_fake + eps))


def adversarial_loss(d_fake: torch.Tensor, eps: float) -> torch.Tensor:
    """Calculate adversarial loss."""
    return -torch.mean(torch.log(d_fake + eps))


class OnlineTripletLoss(nn.Module):
    """
    Online Triplet Loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using a triplet_selector object that takes embeddings and targets and returns indices of
    triplets.
    """

    def __init__(self, margin: float, triplet_selector) -> None:
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Forward pass for OnlineTripletLoss."""
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        # Move triplets to the same device as embeddings
        device = embeddings.device
        triplets = triplets.to(device)

        # Compute positive and negative distances
        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)

        # Calculate losses
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)
