import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from torch.nn import functional as F

from vertical_fl.model import CLIPServerModel

class CLIPFederatedStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.server_model = CLIPServerModel().to(self.device)
        self.optimizer = torch.optim.AdamW(self.server_model.parameters(), lr=5e-5)

    def aggregate_fit(self, rnd, results, failures):
        if not self.accept_failures and failures:
            return None, {}
            
        embedding_results = {}
        
        for _, fit_res in results:
            client_type = fit_res.metrics["client-type"]
            embedding_results[client_type] = torch.from_numpy(parameters_to_ndarrays(fit_res.parameters)[0]).to(self.device)
            
        if len(embedding_results) != 2:
            return None, {"error": "Need both text and image clients to participate"}

        image_embeddings = embedding_results["image"]  
        text_embeddings = embedding_results["text"]   
        
        image_embeddings = image_embeddings.detach().requires_grad_()
        text_embeddings = text_embeddings.detach().requires_grad_()
        
        logits_per_image, logits_per_text = self.server_model(image_embeddings, text_embeddings)
        
        batch_size = image_embeddings.size(0)
        labels = torch.arange(batch_size).long().to(self.device)
        
        loss_img = F.cross_entropy(logits_per_image, labels)
        loss_txt = F.cross_entropy(logits_per_text, labels)
        loss = (loss_img + loss_txt) / 2.0
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        image_grads = image_embeddings.grad.detach().cpu()
        text_grads = text_embeddings.grad.detach().cpu()
        
        parameters_aggregated = ndarrays_to_parameters([
            image_grads.numpy(), 
            text_grads.numpy()
        ])
        
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