from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, logger
from transformers import CLIPProcessor
from vertical_fl.model import CLIPTextClient, CLIPImageClient
from vertical_fl.data_loader import load_fixed_data, load_datasets
from logging import INFO
from dataclasses import dataclass
import torch
import functools # NOT IDEAL WTF
import lovely_tensors as lt; lt.monkey_patch()

_CACHED_IMAGE_DATA = None
_CACHED_TEXT_DATA = None

@functools.lru_cache(maxsize=1)
def get_fixed_data():
    return load_fixed_data()

_CACHED_IMAGE_LOADER = None
_CACHED_TEXT_LOADER = None

def get_datasets(batch_size: int=16):
    train_image_loader, train_text_loader = load_datasets(0, batch_size)
    return iter(train_image_loader), iter(train_text_loader)

@dataclass
class ConfigClient:
    lr: float
    batch_size: int

    aggregate_strategy: str
    use_fixed_data: bool

class TextFlowerClient(NumPyClient):
    def __init__(self, train_text, partition_id: int, config: ConfigClient):
        super().__init__()
        self.properties = {"client_type" : "text"}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = train_text
        self.partition_id = partition_id
        self.config = config
        self.model_text = CLIPTextClient().to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", do_rescale=False)
        self.optimizer = torch.optim.AdamW(self.model_text.parameters(), lr=config.lr)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model_text.state_dict().items()]

    def fit(self, parameters, config):
        text_inputs = self.processor(text=self.data, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_embeddings = self.model_text(**text_inputs)
        return [text_embeddings.detach().cpu().numpy()], len(self.data), {"client-type": "text"}

    def evaluate(self, parameters, config):
        self.model_text.train()
        self.model_text.zero_grad()
        text_inputs = self.processor(text=self.data, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        text_embeddings = self.model_text(**text_inputs)

        grad_tensor = torch.from_numpy(parameters[self.partition_id]).to(self.device)
        text_embeddings.backward(grad_tensor)
        self.optimizer.step()

        return 0.0, len(self.data), {}

class ImageFlowerClient(NumPyClient):
    def __init__(self, train_image, partition_id: int, config: ConfigClient):
        super().__init__()
        self.properties = {"client_type" : "image"}
        self.data = train_image
        self.partition_id = partition_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_image = CLIPImageClient().to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", do_rescale=False)
        self.optimizer = torch.optim.AdamW(self.model_image.parameters(), lr=config.lr)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model_image.state_dict().items()]

    def fit(self, parameters, config):
        image_inputs = self.processor(images=self.data, return_tensors="pt")
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
        with torch.no_grad():
            image_embeddings = self.model_image(**image_inputs)
        return [image_embeddings.detach().cpu().numpy()], len(self.data), {"client-type": "image"}

    def evaluate(self, parameters, config):
        self.model_image.train()
        self.model_image.zero_grad()

        image_inputs = self.processor(images=self.data, return_tensors="pt")
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
        image_embeddings = self.model_image(**image_inputs)

        grad_tensor = torch.from_numpy(parameters[self.partition_id]).to(self.device)
        image_embeddings.backward(grad_tensor)

        self.optimizer.step()

        return 0.0, len(self.data), {}


def client_fn(context: Context):
    """Create either an image or text client based on configuration."""
    # Text client is the odd one
    # Image client is the even one
    assert context.node_config["num-partitions"] % 2 == 0, "Num of client must be even!"

    partition_id = context.node_config.get("partition-id", 0)
    client_type = context.node_config.get("client-type", "image" if partition_id % 2 == 0 else "text")
    aggregate_strategies = ["gather", "reduce"]
    
    config = ConfigClient(
        lr=context.run_config.get("train.learning-rate", 1e-4),
        batch_size=context.run_config.get("train.batch-size", 16),
        use_fixed_data=context.run_config.get("use-fixed-data", True),
        aggregate_strategy=context.node_config.get("aggregate-strategy", "reduce"),
    )

    assert config.aggregate_strategy in aggregate_strategies, f"Aggregate strategy '{config.aggregate_strategy}' is not supported"

    if config.use_fixed_data:
        global _CACHED_IMAGE_DATA, _CACHED_TEXT_DATA
        
        if _CACHED_IMAGE_DATA is None or _CACHED_TEXT_DATA is None:
            _CACHED_IMAGE_DATA, _CACHED_TEXT_DATA = get_fixed_data()
        if client_type == "image":
            return ImageFlowerClient(_CACHED_IMAGE_DATA, partition_id, config).to_client()
        elif client_type == "text":
            return TextFlowerClient(_CACHED_TEXT_DATA, partition_id, config).to_client
    else:
        global _CACHED_IMAGE_LOADER, _CACHED_TEXT_LOADER
        
        if _CACHED_IMAGE_LOADER is None or _CACHED_TEXT_LOADER is None:
            _CACHED_IMAGE_LOADER, _CACHED_TEXT_LOADER = get_datasets(config.batch_size)

        train_image_iterator = _CACHED_IMAGE_LOADER
        train_text_iterator = _CACHED_TEXT_LOADER
        if client_type == "image":
            train_image = next(train_image_iterator)
            logger.log(INFO, f"{train_image}")
            return ImageFlowerClient(train_image, partition_id, config).to_client()
        elif client_type == "text":
            train_text = next(train_text_iterator)
            return TextFlowerClient(train_text, partition_id, config).to_client()

    raise ValueError(f"Unknown client type: {client_type}")


app = ClientApp(
    client_fn=client_fn,
)
