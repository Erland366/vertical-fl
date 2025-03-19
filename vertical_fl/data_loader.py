import torch

from datasets import load_dataset
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from torchvision import transforms
import numpy as np

def image_collate_fn(batch):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize all images to 224x224
        transforms.ToTensor()
    ])
    images = [transform(item["image"]) for item in batch]
    images = torch.stack(images)

    return images

def load_datasets(partition_id: int, batch_size: int):
    fds = FederatedDataset(dataset="nlphuji/flickr30k", partitioners={"test" : 1})
    partition = fds.load_partition(0)
    partition_image = partition.remove_columns([x for x in partition.column_names if x != "image"])
    partition_text = partition.remove_columns([x for x in partition.column_names if x != "caption"])
    partition_text = partition_text.map(lambda x: {"text" : x["caption"][0]})
    train_image = DataLoader(
        partition_image, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=image_collate_fn
    )
    train_text = DataLoader(
        partition_text["text"], 
        batch_size=batch_size, 
        shuffle=False
    )

    return train_image, train_text

def load_image_data(client_id, batch_size=32):
    # For now let's just make it happen, I think this part will be loaded again and again per round
    # TODO: Use FlowerDataset instead
    dataset = load_dataset("nlphuji/flickr30k", split="test")
    
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)
    np.random.seed(42 + client_id)
    np.random.shuffle(indices)
    
    selected_indices = indices[:batch_size]
    images = []
    
    for idx in selected_indices:
        img = dataset[int(idx)]["image"]
        images.append(img)
        
    return images

def load_text_data(client_id, batch_size=32):
    # For now let's just make it happen, I think this part will be loaded again and again per round
    # TODO: Use FlowerDataset instead
    dataset = load_dataset("nlphuji/flickr30k", split="test")
    
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)
    np.random.seed(42 + client_id)  
    np.random.shuffle(indices)
    
    
    selected_indices = indices[:batch_size]
    captions = []
    
    for idx in selected_indices:
        
        caption_idx = (client_id + idx) % 5
        caption = dataset[int(idx)]["caption"][caption_idx]
        captions.append(caption)
        
    return captions

def load_fixed_data(batch_size: int=4):
    dataset = load_dataset("nlphuji/flickr30k", split="test")

    fixed_indices = list(range(1, batch_size + 1))

    images = []
    captions = []

    for idx in fixed_indices:
        img = dataset[int(idx)]["image"]
        images.append(img)

        caption = dataset[int(idx)]["caption"][0]
        captions.append(caption)

    return images, captions


def test_data_loaders():
    print("Testing image data loader...")
    images = load_image_data(0, batch_size=2)
    print(f"Loaded {len(images)} images")
    
    print("\nTesting text data loader...")
    texts = load_text_data(1, batch_size=2)
    print(f"Loaded {len(texts)} text captions")
    print(f"Sample: {texts[0][:50]}...")
    
if __name__ == "__main__":
    test_data_loaders()