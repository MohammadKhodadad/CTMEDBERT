import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive):
        # Normalize the embeddings
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        
        # Compute cosine similarity matrix (anchor-positive, including self-similarity)
        similarity_matrix = torch.matmul(anchor, positive.T) / self.temperature

        # Create labels: each anchor should be most similar to its corresponding positive
        batch_size = anchor.size(0)
        labels = torch.arange(batch_size).to(anchor.device)

        # Apply cross-entropy loss on the similarity matrix
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss
if __name__ == "__main__":
    # Example usage
    batch_size = 16
    embedding_dim = 128

    anchor = torch.randn(batch_size, embedding_dim)  # Embedding for anchor (sentence 1)
    positive = torch.randn(batch_size, embedding_dim)  # Embedding for positive (sentence 2)

    loss_fn = InfoNCELoss(temperature=0.07)
    loss = loss_fn(anchor, positive)
    print("Modified InfoNCE Loss:", loss.item())
