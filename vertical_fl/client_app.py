from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from sklearn.preprocessing import StandardScaler
from transformers import CLIPProcessor
import torch

from vertical_fl.task import ClientModel, load_data
from vertical_fl.model import CLIPTextClient, CLIPImageClient

class TextFlowerClient(NumPyClient):
    def __init__(self, data, lr: float=1e-4):
        super().__init__()
        self.properties = {"client_type" : "text"}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = data
        self.model = CLIPTextClient().to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
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
    def __init__(self, data, lr: float=1e-4):
        super().__init__()
        self.properties = {"client_type" : "image"}
        self.data = data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPImageClient().to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
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


class FlowerClient(NumPyClient):
    def __init__(self, v_split_id, data, lr):
        self.v_split_id = v_split_id
        self.data = torch.tensor(StandardScaler().fit_transform(data)).float()
        self.model = ClientModel(input_size=self.data.shape[1])
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def get_parameters(self, config):
        pass

    def fit(self, parameters, config):
        embedding = self.model(self.data)
        return [embedding.detach().numpy()], 1, {}

    def evaluate(self, parameters, config):
        self.model.zero_grad()
        embedding = self.model(self.data)
        embedding.backward(torch.from_numpy(parameters[int(self.v_split_id)]))
        self.optimizer.step()
        return 0.0, 1, {}


def client_fn(context: Context):
    """Create either an image or text client based on configuration."""
    partition_id = context.node_config.get("partition-id", 0)
    client_type = context.node_config.get("client-type", "image" if partition_id % 2 == 0 else "text")
    
    lr = context.run_config.get("learning-rate", 1e-4)

    if client_type == "image":
        from vertical_fl.data_loader import load_image_data
        image_data = load_image_data(partition_id)
        return ImageFlowerClient(image_data, lr).to_client()
    
    elif client_type == "text":
        from vertical_fl.data_loader import load_text_data
        text_data = load_text_data(partition_id)
        return TextFlowerClient(text_data, lr).to_client()
    
    else:
        raise ValueError(f"Unknown client type: {client_type}")


app = ClientApp(
    client_fn=client_fn,
)
