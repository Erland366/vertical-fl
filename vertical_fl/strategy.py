import json
import flwr as fl
import torch
import wandb
from logging import INFO
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, logger
from torch.nn import functional as F
from datetime import datetime
from pathlib import Path

from vertical_fl.model import CLIPServerModel

PROJECT_NAME = "VFL-CLIP"

def create_run_dir(config = None) -> Path:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)

    # Save run config as json
    if config is not None:
        with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
            json.dump(config, fp)

    return save_path, run_dir

class CLIPFederatedStrategy(fl.server.strategy.FedAvg):
    def __init__(self, lr: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.server_model = CLIPServerModel().to(self.device)
        self.optimizer = torch.optim.AdamW(self.server_model.parameters(), lr=lr)

        # TODO: Add config here later on
        self.save_path, self.run_dir = create_run_dir()
        self.results = {}

        self._init_wandb_project()

    def _init_wandb_project(self):
        wandb.init(
            project=PROJECT_NAME, 
            name="CLIP-FedAvg", 
            # config=self.config # TODO: Add config here later on
        )

    def _store_results(self, tag: str, results_dict):
        """Store results in dictionary, then save as JSON."""
        # Update results dict
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]

        # Save results to disk.
        # Note we overwrite the same file with each call to this function.
        # While this works, a more sophisticated approach is preferred
        # in situations where the contents to be saved are larger.
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)

    def store_results_and_log(self, server_round: int, tag: str, results_dict):
        """A helper method that stores results and logs them to W&B if enabled."""
        # Store results
        self._store_results(
            tag=tag,
            results_dict={"round": server_round, **results_dict},
        )

            # Log centralized loss and metrics to W&B
        wandb.log(results_dict, step=server_round)

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
        labels = torch.arange(batch_size, device=self.device).long()
        
        loss_img = F.cross_entropy(logits_per_text.t(), labels)
        loss_txt = F.cross_entropy(logits_per_text, labels)
        loss = (loss_img + loss_txt) / 2.0
        logger.log(INFO, f"Round {rnd} loss: {loss.item()}")

        
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
            "loss_img": loss_img.item(),
            "loss_txt": loss_txt.item(),
            "i2t_acc": i2t_acc,
            "t2i_acc": t2i_acc,
            "avg_acc": avg_acc,
        }

        self.store_results_and_log(
            server_round=rnd,
            tag="aggregate_fit",
            results_dict=metrics_aggregated,
        )
        wandb.log(metrics_aggregated, step=rnd)
        
        return parameters_aggregated, metrics_aggregated

    def get_fit_config_fn(self, server_round):
        """Return a function which returns the fit configuration."""
        def fit_config(server_round):
            return {"server_round": server_round}
        return fit_config