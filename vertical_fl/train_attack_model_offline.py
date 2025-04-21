import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import CLIPProcessor
import argparse
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
from datasets import load_dataset 

from vertical_fl.model import CLIPImageClient, CLIPTextClient, CLIPServerModel
from vertical_fl.attack_model import AttackFromImageNetWithGradient, AttackFromTextNetWithGradient

SIDE_DATASET_NAME = "nlphuji/flickr30k"
SIDE_DATASET_SPLIT = "test" 
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20 
DEFAULT_LEARNING_RATE = 3e-4 
CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VFLSimulationDataset(Dataset):
    """
    Loads pairs of image/text data for VFL simulation.
    Processing happens externally for flexibility.
    """
    def __init__(self, dataset_name, dataset_split):
        self.dataset = load_dataset(dataset_name, split=dataset_split)
        print(f"Loaded dataset '{dataset_name}' split '{dataset_split}' with {len(self.dataset)} samples.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[int(idx)]
        image = item["image"]
        
        caption = item["caption"][0] if isinstance(item["caption"], list) else item["caption"]
        return {"image": image, "text": caption}


def collate_fn(batch, processor):
    """Processes a batch of images and text using the CLIP processor."""
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding="max_length", 
        truncation=True,
        max_length=77 
    )
    return inputs



def train_attack_model(args):
    print(f"Starting offline training for attacker: {args.attacker_type}")
    print(f"Using device: {DEVICE}")
    print(f"Side data size: {args.side_data_size}")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}")
    print(f"Loss type: {args.loss_type}")

    
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    image_encoder = CLIPImageClient().to(DEVICE)
    text_encoder = CLIPTextClient().to(DEVICE)
    server_model = CLIPServerModel().to(DEVICE)

    
    for param in image_encoder.parameters():
        param.requires_grad = False
    for param in text_encoder.parameters():
        param.requires_grad = False
    for param in server_model.parameters():
        param.requires_grad = False

    image_encoder.eval()
    text_encoder.eval()
    server_model.eval()
    print("Loaded and froze VFL simulation models.")

    
    full_side_dataset = VFLSimulationDataset(SIDE_DATASET_NAME, SIDE_DATASET_SPLIT)

    if args.side_data_size is not None and args.side_data_size < len(full_side_dataset):
        
        
        indices = np.random.choice(len(full_side_dataset), args.side_data_size, replace=False)
        side_dataset = Subset(full_side_dataset, indices)
        print(f"Using a subset of {args.side_data_size} samples for training.")
    else:
        side_dataset = full_side_dataset
        print(f"Using the full dataset split ({len(side_dataset)} samples) for training.")

    
    dataloader = DataLoader(
        side_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor) 
    )

    
    if args.attacker_type == 'image':
        attack_model = AttackFromImageNetWithGradient(
            image_embedding_dim=512, 
            gradient_dim=512,        
            text_embedding_dim=512
        ).to(DEVICE)
        target_description = "Text Embeddings"
    elif args.attacker_type == 'text':
        attack_model = AttackFromTextNetWithGradient(
            text_embedding_dim=512,
            gradient_dim=512,
            image_embedding_dim=512
        ).to(DEVICE)
        target_description = "Image Embeddings"
    else:
        raise ValueError("Invalid attacker_type. Choose 'image' or 'text'.")

    optimizer = torch.optim.AdamW(attack_model.parameters(), lr=args.lr)

    
    if args.loss_type == 'mse':
        criterion = nn.MSELoss()
    elif args.loss_type == 'cosine':
        criterion = nn.CosineEmbeddingLoss(margin=0.1) 
    else:
        raise ValueError("Invalid loss_type. Choose 'mse' or 'cosine'.")

    print(f"Initialized attack model ({type(attack_model).__name__}) targeting {target_description}.")
    print(f"Optimizer: AdamW, Criterion: {type(criterion).__name__}")


    
    attack_model.train() 
    for epoch in range(args.epochs):
        total_epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)

        for batch_inputs in progress_bar:
            
            batch_inputs = {k: v.to(DEVICE) for k, v in batch_inputs.items()}
            current_batch_size = batch_inputs['pixel_values'].shape[0] if 'pixel_values' in batch_inputs else batch_inputs['input_ids'].shape[0]

            
            optimizer.zero_grad() 

            
            with torch.no_grad(): 
                image_embeddings = image_encoder(pixel_values=batch_inputs['pixel_values'])
                text_embeddings = text_encoder(
                    input_ids=batch_inputs['input_ids'],
                    attention_mask=batch_inputs['attention_mask']
                )

            
            image_embeddings_server = image_embeddings.detach().clone().requires_grad_(True)
            text_embeddings_server = text_embeddings.detach().clone().requires_grad_(True)

            
            logits_per_image, logits_per_text = server_model(image_embeddings_server, text_embeddings_server)
            labels = torch.arange(current_batch_size, device=DEVICE).long()

            
            loss_img = F.cross_entropy(logits_per_image, labels)
            loss_txt = F.cross_entropy(logits_per_text, labels)
            vfl_loss = (loss_img + loss_txt) / 2.0

            
            vfl_loss.backward()

            
            
            server_grad_image = image_embeddings_server.grad.detach().clone() if image_embeddings_server.grad is not None else torch.zeros_like(image_embeddings_server)
            server_grad_text = text_embeddings_server.grad.detach().clone() if text_embeddings_server.grad is not None else torch.zeros_like(text_embeddings_server)

            

            
            if args.attacker_type == 'image':
                
                attack_input_emb = image_embeddings.detach() 
                attack_input_grad = server_grad_image
                target_embedding = text_embeddings.detach() 
                
                predicted_embedding = attack_model(attack_input_emb, attack_input_grad)

            else: 
                
                attack_input_emb = text_embeddings.detach()
                attack_input_grad = server_grad_text
                target_embedding = image_embeddings.detach() 
                
                predicted_embedding = attack_model(attack_input_emb, attack_input_grad)

            
            if args.loss_type == 'mse':
                attack_loss = criterion(predicted_embedding, target_embedding)
            elif args.loss_type == 'cosine':
                
                cosine_target = torch.ones(current_batch_size).to(DEVICE)
                attack_loss = criterion(predicted_embedding, target_embedding, cosine_target)

            
            attack_loss.backward()
            optimizer.step()

            total_epoch_loss += attack_loss.item()
            progress_bar.set_postfix({'loss': attack_loss.item()})

      
        avg_epoch_loss = total_epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs} - Average Attack Loss: {avg_epoch_loss:.6f}")

    
    output_dir = Path(f"attack_models_{args.attacker_type}_gradient")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_filename = f"attack_model_sidesize_{args.side_data_size}_loss_{args.loss_type}.pth"
    model_save_path = output_dir / model_filename
    torch.save(attack_model.state_dict(), model_save_path)
    print(f"Attack model saved to: {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Training for VFL Attack Models with Gradients")
    parser.add_argument('--attacker_type', type=str, required=True, choices=['image', 'text'],
                        help="Which client is the attacker ('image' or 'text').")
    parser.add_argument('--side_data_size', type=int, default=None,
                        help="Number of samples from side dataset to use (default: all).")
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size.")
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate.")
    parser.add_argument('--loss_type', type=str, default='cosine', choices=['mse', 'cosine'],
                        help="Loss function for attack model ('mse' or 'cosine').")

    args = parser.parse_args()
    train_attack_model(args)