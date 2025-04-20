import torch
import torch.nn as nn

class AttackFromTextNet(nn.Module):
    def __init__(self, text_embedding_dim=512, image_embedding_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(text_embedding_dim, text_embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(text_embedding_dim * 2, image_embedding_dim),
        )

    def forward(self, text_embeddings):
        return self.mlp(text_embeddings)