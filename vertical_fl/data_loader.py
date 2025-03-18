from datasets import load_dataset
import numpy as np

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