import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from torch.nn import functional as F

from vertical_fl.model import CLIPServerModel


class ServerModel(nn.Module):
    def __init__(self, input_size):
        super(ServerModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        return self.sigmoid(x)

class CLIPFederatedStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.server_model = CLIPServerModel()
        self.optimizer = torch.optim.AdamW(self.server_model.parameters(), lr=5e-5)

    def aggregate_fit(self, rnd, results, failures):
        if not self.accept_failures and failures:
            return None, {}
            
        # Get client embeddings
        embedding_results = {}
        original_shapes = {}
        
        for _, fit_res in results:
            client_type = fit_res.metrics["client-type"]
            embedding_results[client_type] = torch.from_numpy(parameters_to_ndarrays(fit_res.parameters)[0])
            # Store original shapes to properly reshape gradients later
            original_shapes[client_type] = embedding_results[client_type].shape
            
        if len(embedding_results) != 2:
            return None, {"error": "Need both text and image clients to participate"}

        # Get embeddings
        image_embeddings = embedding_results["image"]  
        text_embeddings = embedding_results["text"]   
        
        # Set requires grad for backward pass
        image_embeddings = image_embeddings.detach().requires_grad_()
        text_embeddings = text_embeddings.detach().requires_grad_()
        
        # Forward pass through server model
        logits_per_image, logits_per_text = self.server_model(image_embeddings, text_embeddings)
        
        # Compute loss
        batch_size = image_embeddings.size(0)
        labels = torch.arange(batch_size).long().to(image_embeddings.device)
        
        loss_img = F.cross_entropy(logits_per_image, labels)
        loss_txt = F.cross_entropy(logits_per_text, labels)
        loss = (loss_img + loss_txt) / 2.0
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Get gradients and process them through the inverse projection if needed
        image_grads = image_embeddings.grad.detach()
        text_grads = text_embeddings.grad.detach()
        
        # Pack gradients and return
        parameters_aggregated = ndarrays_to_parameters([
            image_grads.numpy(), 
            text_grads.numpy()
        ])
        
        # Calculate metrics
        with torch.no_grad():
            i2t_pred = logits_per_image.argmax(dim=1)
            i2t_acc = (i2t_pred == labels).float().mean().item() * 100
            
            t2i_pred = logits_per_text.argmax(dim=1)
            t2i_acc = (t2i_pred == labels).float().mean().item() * 100
            
            avg_acc = (i2t_acc + t2i_acc) / 2
        
        metrics_aggregated = {
            "loss": loss.item(),
            "i2t_acc": i2t_acc,
            "t2i_acc": t2i_acc,
            "avg_acc": avg_acc,
        }
        
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(self, rnd, results, failures):
        # For CLIP, evaluation happens during training via contrastive loss
        # But we could implement a separate evaluation if needed
        return None, {}

    def get_fit_config_fn(self, server_round):
        """Return a function which returns the fit configuration."""
        def fit_config(server_round):
            return {"server_round": server_round}
        return fit_config
            

class Strategy(fl.server.strategy.FedAvg):
    def __init__(self, labels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = ServerModel(12)
        self.initial_parameters = ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.BCELoss()
        self.label = torch.tensor(labels).float().unsqueeze(1)

    def aggregate_fit(
        self,
        rnd,
        results,
        failures,
    ):
        
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        embedding_results = [
            torch.from_numpy(parameters_to_ndarrays(fit_res.parameters)[0])
            for _, fit_res in results
        ]
        embeddings_aggregated = torch.cat(embedding_results, dim=1)
        embedding_server = embeddings_aggregated.detach().requires_grad_()
        output = self.model(embedding_server)
        loss = self.criterion(output, self.label)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        grads = embedding_server.grad.split([4, 4, 4], dim=1)
        np_grads = [grad.numpy() for grad in grads]
        parameters_aggregated = ndarrays_to_parameters(np_grads)

        with torch.no_grad():
            correct = 0
            output = self.model(embedding_server)
            predicted = (output > 0.5).float()

            correct += (predicted == self.label).sum().item()

            accuracy = correct / len(self.label) * 100

        metrics_aggregated = {"accuracy": accuracy}

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        rnd,
        results,
        failures,
    ):
        return None, {}