from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, logger
from transformers import CLIPProcessor
from vertical_fl.model import CLIPTextClient, CLIPImageClient
from vertical_fl.data_loader import load_fixed_data, load_datasets
from vertical_fl.attack_model import AttackNet
from logging import INFO
from dataclasses import dataclass
import os
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
class ConfigClientAttack:
    lr: float
    batch_size: int
    aggregate_strategy: str
    use_fixed_data: bool

    log_attack_metrics: bool = True
    attack_model_path: str = None # Path to load the pre-trained attack model
    attack_side_data_size: int = 0 # Number of samples to use for attack evaluation


class TextFlowerClientAttacker(NumPyClient):
    def __init__(self, train_text, partition_id: int, config: ConfigClientAttack):
        super().__init__()
        self.properties = {"client_type" : "text"}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = train_text
        self.partition_id = partition_id
        self.config = config
        self.model_text = CLIPTextClient().to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", do_rescale=False)
        self.optimizer = torch.optim.AdamW(self.model_text.parameters(), lr=config.lr)

        self.attack_model = None
        if self.config.attack_model_path and os.path.exists(self.config.attack_model_path):
            self.attack_model = AttackNet().to(self.device)
            self.attack_model.load_state_dict(torch.load(self.config.attack_model_path, map_location=self.device))
            self.attack_model.eval() # Attack model should be in eval mode during FL
            logger.log(INFO, f"Text client {partition_id} loaded attack model from {self.config.attack_model_path}")
        else:
            logger.log(INFO, f"Text client {partition_id}: Attack model not loaded (path not specified or not found).")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model_text.state_dict().items()]

    def fit(self, parameters, config):
        self.model_text.train()

        text_inputs = self.processor(text=self.data, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        with torch.no_grad():
             text_embeddings = self.model_text(**text_inputs)

        predicted_image_embeddings = None
        if self.attack_model is not None:
             with torch.no_grad():
                 predicted_image_embeddings = self.attack_model(text_embeddings)
             logger.log(INFO, f"Text client {self.partition_id} predicted image embeddings.")

        # The server expects a list of numpy arrays as parameters
        # We send the text embeddings and the predicted image embeddings (if attack is enabled)
        # THIS IS ONLY FOR EVALUATION
        params_to_send = [text_embeddings.detach().cpu().numpy()]
        if predicted_image_embeddings is not None:
            params_to_send.append(predicted_image_embeddings.detach().cpu().numpy())

        num_examples = len(self.data)

        metrics = {"client-type": "text"}

        return params_to_send, num_examples, metrics

    def evaluate(self, parameters, config):
        self.model_text.train()
        self.model_text.zero_grad()
        with torch.enable_grad():
            text_inputs = self.processor(text=self.data, return_tensors="pt", padding=True, truncation=True)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            text_embeddings = self.model_text(**text_inputs)

        if len(parameters) <= self.partition_id:
             logger.log(WARN, f"Client {self.partition_id}: Expected gradient at index {self.partition_id} but parameters list has length {len(parameters)}. Skipping update.")
             return 0.0, len(self.data), {}

        grad_tensor_correct_idx = torch.from_numpy(parameters[self.partition_id]).to(self.device)

        if text_embeddings.requires_grad:
             text_embeddings.backward(grad_tensor_correct_idx)
             for name, param in self.model_text.named_parameters():
                 if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                     logger.log(WARN, f"Client {self.partition_id}: NaN/Inf gradient detected in {name}. Skipping optimizer step.")
                     return 0.0, len(self.data), {} 


             self.optimizer.step()
        else:
             logger.log(WARN, f"Client {self.partition_id}: Text embeddings do not require grad. Cannot apply received gradient. Skipping update.")

        return 0.0, len(self.data), {}


class ImageFlowerClient(NumPyClient):
    def __init__(self, train_image, partition_id: int, config: ConfigClientAttack):
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
    
    config = ConfigClientAttack(
        lr=context.run_config.get("train.learning-rate", 1e-4),
        batch_size=context.run_config.get("train.batch-size", 16),
        use_fixed_data=context.run_config.get("use-fixed-data", True),
        aggregate_strategy=context.node_config.get("aggregate-strategy", "reduce"),
        log_attack_metrics=context.run_config.get("log.log-attack-metrics", False),
        attack_model_path=context.run_config.get("attack.model_path", None),
        attack_side_data_size=context.run_config.get("attack.side_data_size", 0),
    )

    assert config.aggregate_strategy in aggregate_strategies, f"Aggregate strategy '{config.aggregate_strategy}' is not supported"

    if config.use_fixed_data:
        global _CACHED_IMAGE_DATA, _CACHED_TEXT_DATA
        
        if _CACHED_IMAGE_DATA is None or _CACHED_TEXT_DATA is None:
            _CACHED_IMAGE_DATA, _CACHED_TEXT_DATA = get_fixed_data()
        if client_type == "image":
            return ImageFlowerClient(_CACHED_IMAGE_DATA, partition_id, config).to_client()
        elif client_type == "text":
            return TextFlowerClientAttacker(_CACHED_TEXT_DATA, partition_id, config).to_client
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
            return TextFlowerClientAttacker(train_text, partition_id, config).to_client()

    raise ValueError(f"Unknown client type: {client_type}")


app = ClientApp(
    client_fn=client_fn,
)
