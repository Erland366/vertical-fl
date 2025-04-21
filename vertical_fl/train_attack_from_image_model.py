import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel
from vertical_fl.attack_model import AttackFromTextNet
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import os

SIDE_DATASET_NAME = "nlphuji/flickr30k"
SIDE_DATASET_SPLIT = "test" 
MODEL_SAVE_PATH = "attack_model_weights.pth"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SideDataset(Dataset):
    def __init__(self, dataset_split, processor):
        self.dataset = load_dataset(SIDE_DATASET_NAME, split=dataset_split)
        self.processor = processor
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(DEVICE)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        caption = item["caption"][0] # Use the first caption

        # Generate embeddings using the pre-trained CLIP model
        # This mimics what the respective client encoders would produce *if* they were frozen
        # For simplicity, we use the full CLIP model here to get reference embeddings
        with torch.no_grad():
            image_inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)
            text_inputs = self.processor(text=caption, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            image_features = self.clip_model.get_image_features(**image_inputs)
            text_features = self.clip_model.get_text_features(**text_inputs)

        return text_features.squeeze().cpu().numpy(), image_features.squeeze().cpu().numpy() # Remove batch dim


def main(side_data_size=None):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", do_rescale=False)
    side_dataset = SideDataset(SIDE_DATASET_SPLIT, processor)

    if side_data_size is not None and side_data_size < len(side_dataset):
        indices = list(range(side_data_size))
        side_dataset = torch.utils.data.Subset(side_dataset, indices)


    dataloader = DataLoader(side_dataset, batch_size=BATCH_SIZE, shuffle=True)

    attack_net = AttackFromTextNet().to(DEVICE)
    optimizer = torch.optim.AdamW(attack_net.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() # Or nn.CosineSimilarity loss, inverted

    print(f"Training AttackNet on {len(side_dataset)} samples...")
    attack_net.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for text_embs, image_embs in dataloader:
            text_embs, image_embs = text_embs.to(DEVICE), image_embs.to(DEVICE)

            optimizer.zero_grad()
            predicted_text_embs = attack_net(image_embs)
            loss = criterion(predicted_text_embs, text_embs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(dataloader):.4f}")

    os.makedirs("attack_models_from_image", exist_ok=True)
    model_save_path = os.path.join("attack_models_from_image", f"attack_model_sidesize_{side_data_size}.pth")
    torch.save(attack_net.state_dict(), model_save_path)
    print(f"Attack model saved to {model_save_path}")

if __name__ == "__main__":
    main(side_data_size=10)
    main(side_data_size=100)
    main(side_data_size=250)
    main(side_data_size=500)
    main(side_data_size=1000)