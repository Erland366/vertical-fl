import torch
import torch.nn as nn

class AttackNet(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        # Simple MLP to map text embeddings to predicted image embeddings
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, text_embeddings):
        return self.mlp(text_embeddings)