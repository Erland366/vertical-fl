import torch
import torch.nn as nn

class AttackFromTextNet(nn.Module):
    """Predicts image embedding from text embedding only."""
    def __init__(self, text_embedding_dim=512, image_embedding_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(text_embedding_dim, text_embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(text_embedding_dim * 2, image_embedding_dim),
        )

    def forward(self, text_embeddings):
        return self.mlp(text_embeddings)

class AttackFromImageNet(nn.Module):
    """Predicts text embedding from image embedding only."""
    def __init__(self, text_embedding_dim=512, image_embedding_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(image_embedding_dim, image_embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(image_embedding_dim * 2, text_embedding_dim),
        )

    def forward(self, image_embeddings):
        return self.mlp(image_embeddings)

class AttackFromTextNetWithGradient(nn.Module):
    """
    Predicts image embedding using the text client's own text embedding
    and the gradient received from the server.
    """
    def __init__(self, text_embedding_dim=512, gradient_dim=512, image_embedding_dim=512):
        super().__init__()
        input_dim = text_embedding_dim + gradient_dim
        hidden_dim = (input_dim + image_embedding_dim) // 2 

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2), 
            nn.BatchNorm1d(hidden_dim * 2),       
            nn.ReLU(),
            nn.Dropout(0.3),                      
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),           
            nn.ReLU(),
            nn.Dropout(0.3),                      
            nn.Linear(hidden_dim, image_embedding_dim),
        )

    def forward(self, text_embeddings, gradients):
        combined_input = torch.cat((text_embeddings, gradients), dim=1)
        return self.mlp(combined_input)

class AttackFromImageNetWithGradient(nn.Module):
    """
    Predicts text embedding using the image client's own image embedding
    and the gradient received from the server.
    """
    def __init__(self, image_embedding_dim=512, gradient_dim=512, text_embedding_dim=512):
        super().__init__()
        input_dim = image_embedding_dim + gradient_dim
        hidden_dim = (input_dim + text_embedding_dim) // 2 

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2), 
            nn.BatchNorm1d(hidden_dim * 2),       
            nn.ReLU(),
            nn.Dropout(0.3),                      
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),           
            nn.ReLU(),
            nn.Dropout(0.3),                      
            nn.Linear(hidden_dim, text_embedding_dim),
        )

    def forward(self, image_embeddings, gradients):
        combined_input = torch.cat((image_embeddings, gradients), dim=1)
        return self.mlp(combined_input)

def test_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    text_emb_dim = 512
    img_emb_dim = 512
    grad_dim = 512

    text_attacker = AttackFromTextNetWithGradient(text_emb_dim, grad_dim, img_emb_dim).to(device)
    dummy_text_embs = torch.randn(batch_size, text_emb_dim).to(device)
    dummy_grads_text = torch.randn(batch_size, grad_dim).to(device)
    predicted_img_embs = text_attacker(dummy_text_embs, dummy_grads_text)
    print(f"AttackFromTextNetWithGradient input shapes: {dummy_text_embs.shape}, {dummy_grads_text.shape}")
    print(f"AttackFromTextNetWithGradient output shape: {predicted_img_embs.shape}")
    assert predicted_img_embs.shape == (batch_size, img_emb_dim)

    image_attacker = AttackFromImageNetWithGradient(img_emb_dim, grad_dim, text_emb_dim).to(device)
    dummy_img_embs = torch.randn(batch_size, img_emb_dim).to(device)
    dummy_grads_img = torch.randn(batch_size, grad_dim).to(device)
    predicted_text_embs = image_attacker(dummy_img_embs, dummy_grads_img)
    print(f"\nAttackFromImageNetWithGradient input shapes: {dummy_img_embs.shape}, {dummy_grads_img.shape}")
    print(f"AttackFromImageNetWithGradient output shape: {predicted_text_embs.shape}")
    assert predicted_text_embs.shape == (batch_size, text_emb_dim)


if __name__ == "__main__":
    test_models()