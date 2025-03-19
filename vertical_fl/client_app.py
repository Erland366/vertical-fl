from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from transformers import CLIPProcessor
from vertical_fl.model import CLIPTextClient, CLIPImageClient
from vertical_fl.data_loader import load_fixed_data, load_datasets
import torch
import functools # NOT IDEAL WTF

_CACHED_IMAGE_DATA = None
_CACHED_TEXT_DATA = None

@functools.lru_cache(maxsize=1)
def get_fixed_data():
    from vertical_fl.data_loader import load_fixed_data
    return load_fixed_data()

_CACHED_IMAGE_LOADER = None
_CACHED_TEXT_LOADER = None

@functools.lru_cache(maxsize=1)
def get_datasets():
    train_image_loader, train_text_loader = load_datasets(0, 32)
    return train_image_loader, train_text_loader


class TextFlowerClient(NumPyClient):
    def __init__(self, train_text, lr: float=1e-4):
        super().__init__()
        self.properties = {"client_type" : "text"}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = train_text
        self.model = CLIPTextClient().to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", do_rescale=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        text_inputs = self.processor(text=self.data, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_embeddings = self.model(**text_inputs)
        return [text_embeddings.detach().numpy()], len(self.data), {"client-type": "text"}

    def evaluate(self, parameters, config):
        self.model.zero_grad()
        text_inputs = self.processor(text=self.data, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        text_embeddings = self.model(**text_inputs)

        grad_tensor = torch.from_numpy(parameters[1]).to(self.device)
        text_embeddings.backward(grad_tensor)
        self.optimizer.step()

        return 0.0, len(self.data), {}

class ImageFlowerClient(NumPyClient):
    def __init__(self, train_image, lr: float=1e-4):
        super().__init__()
        self.properties = {"client_type" : "image"}
        self.data = train_image
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPImageClient().to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", do_rescale=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        image_inputs = self.processor(images=self.data, return_tensors="pt")
        with torch.no_grad():
            image_embeddings = self.model(**image_inputs)
        return [image_embeddings.detach().numpy()], len(self.data), {"client-type": "image"}

    def evaluate(self, parameters, config):
        self.model.zero_grad()

        image_inputs = self.processor(images=self.data, return_tensors="pt")
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
        image_embeddings = self.model(**image_inputs)

        grad_tensor = torch.from_numpy(parameters[0]).to(self.device)
        image_embeddings.backward(grad_tensor)
        self.optimizer.step()

        return 0.0, len(self.data), {}


def client_fn(context: Context):
    """Create either an image or text client based on configuration."""
    partition_id = context.node_config.get("partition-id", 0)
    client_type = context.node_config.get("client-type", "image" if partition_id % 2 == 0 else "text")

    use_fixed_data = context.run_config.get("use-fixed-data", True)
    
    lr = context.run_config.get("learning-rate", 1e-4)

    if use_fixed_data:
        global _CACHED_IMAGE_DATA, _CACHED_TEXT_DATA
        
        if _CACHED_IMAGE_DATA is None or _CACHED_TEXT_DATA is None:
            _CACHED_IMAGE_DATA, _CACHED_TEXT_DATA = get_fixed_data()
        if client_type == "image":
            return ImageFlowerClient(_CACHED_IMAGE_DATA, lr).to_client()
        elif client_type == "text":
            return TextFlowerClient(_CACHED_TEXT_DATA, lr).to_client
    else:
        global _CACHED_IMAGE_LOADER, _CACHED_TEXT_LOADER
        
        if _CACHED_IMAGE_LOADER is None or _CACHED_TEXT_LOADER is None:
            _CACHED_IMAGE_LOADER, _CACHED_TEXT_LOADER = get_fixed_data()

        train_image = next(iter(_CACHED_IMAGE_LOADER))
        train_text = next(iter(_CACHED_TEXT_LOADER))
        if client_type == "image":
            return ImageFlowerClient(train_image, lr).to_client()
        elif client_type == "text":
            return TextFlowerClient(train_text, lr).to_client()

    raise ValueError(f"Unknown client type: {client_type}")


app = ClientApp(
    client_fn=client_fn,
)
