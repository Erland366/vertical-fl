import json
import flwr as fl
import torch
import wandb
from logging import INFO, WARN
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, logger
from torch.nn import functional as F
from datetime import datetime
from pathlib import Path

from vertical_fl.model import CLIPServerModel
from dataclasses import dataclass, field

def create_run_dir(config = None) -> Path:
    """Create a directory where to save results from this run."""
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)

    if config is not None:
        with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
            json.dump(config, fp)

    return save_path, run_dir

@dataclass
class ConfigServerAttackGradient:
    lr: float
    num_rounds: int
    batch_size: int
    use_fixed_data: bool
    aggregate_strategy: str
    num_partitions: int | None = None
    project_name: str = "VFL-CLIP-Attack"
    run_name: str = "CLIP-FedAvg"

    log_attack_metrics: bool = True 
    attack_model_path: str = None 
    attack_side_data_size: int = 0 
    whos_attacking: str = "text" 

    attack_active_online: bool = True 
    log_attack_predictions_client: bool = True 
    log_ground_truth_server: bool = True 

    wandb_project: str | None = field(init=False) 

    def __post_init__(self):
        self.wandb_project = self.project_name 

class CLIPFederatedStrategyAttackGradient(fl.server.strategy.FedAvg):
    def __init__(self, config: ConfigServerAttackGradient, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.server_model = CLIPServerModel().to(self.device)
        self.config = config 
        self.optimizer = torch.optim.AdamW(self.server_model.parameters(), lr=config.lr)

        self.save_path, self.run_dir = create_run_dir()
        self.results = {}

        self._init_wandb_project()

    def _init_wandb_project(self):
        run_name = (
            self.config.run_name.format(size=self.config.attack_side_data_size) +
            f"_{self.config.aggregate_strategy}" + 
            f"_p{self.config.num_partitions}" + 
            f"_r{self.config.num_rounds}" +
            f"_attack-{self.config.whos_attacking}"
        )
        wandb.init(
            project=self.config.wandb_project, 
            name=run_name,
            config=vars(self.config) 
        )
        wandb.config.update(vars(self.config), allow_val_change=True)

    def _store_results(self, tag: str, results_dict):
        """Store results in dictionary, then save as JSON."""
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]

        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)

    def store_results_and_log(self, server_round: int, tag: str, results_dict):
        """A helper method that stores results and logs them to W&B if enabled."""
        self._store_results(
            tag=tag,
            results_dict={"round": server_round, **results_dict},
        )
        wandb.log(results_dict, step=server_round)


    def aggregate_fit(
        self,
        rnd: int,
        results,
        failures,
    ):
        """Aggregate fit results and log ground truth if configured."""
        self.config.num_partitions = len(results)
        if self.config.num_partitions % 2 != 0:
             logger.log(WARN, f"Round {rnd}: Uneven number of clients ({self.config.num_partitions}). VFL pairing might be broken.")

        if not self.accept_failures and failures:
            return None, {}

        image_client_results = []
        text_client_results = []
        for client, fit_res in results:
            client_type = fit_res.metrics.get("client-type")
            params = parameters_to_ndarrays(fit_res.parameters)
            if not params:
                 logger.log(WARN, f"Client {client.cid} sent empty parameters. Skipping.")
                 continue
            embedding_batch = params[0] 

            if client_type == "image":
                image_client_results.append((client.cid, embedding_batch))
            elif client_type == "text":
                text_client_results.append((client.cid, embedding_batch))
            else:
                logger.log(WARN, f"Client {client.cid}: Unknown client type '{client_type}'. Skipping.")

        if not image_client_results or not text_client_results:
            logger.log(WARN, f"Round {rnd}: Missing image or text client results. Cannot perform VFL.")
            return None, {"error": "Missing image or text client results"}

        
        sum_text_embeddings = sum([emb for _, emb in text_client_results])
        sum_image_embeddings = sum([emb for _, emb in image_client_results])
        num_image_clients = len(image_client_results)
        num_text_clients = len(text_client_results)

        avg_image_embeddings = torch.from_numpy(sum_image_embeddings / num_image_clients).to(self.device)
        avg_text_embeddings = torch.from_numpy(sum_text_embeddings / num_text_clients).to(self.device)

        
        if self.config.log_ground_truth_server and wandb.run is not None:
            try:
                log_data = {
                    "server_ground_truth/avg_image_embedding_mean": avg_image_embeddings.mean().item(),
                    "server_ground_truth/avg_image_embedding_std": avg_image_embeddings.std().item(),
                    "server_ground_truth/avg_text_embedding_mean": avg_text_embeddings.mean().item(),
                    "server_ground_truth/avg_text_embedding_std": avg_text_embeddings.std().item(),
                }
                wandb.log(log_data, step=rnd)
            except Exception as e:
                logger.log(WARN, f"Round {rnd}: Failed to log ground truth embeddings to WandB: {e}")

        avg_image_embeddings.requires_grad_(True)
        avg_text_embeddings.requires_grad_(True)

        logits_per_image, logits_per_text = self.server_model(avg_image_embeddings, avg_text_embeddings)

        batch_size = avg_image_embeddings.size(0)
        labels = torch.arange(batch_size, device=self.device).long()

        loss_img = F.cross_entropy(logits_per_text.t(), labels)
        loss_txt = F.cross_entropy(logits_per_text, labels)
        vfl_loss = (loss_img + loss_txt) / 2.0

        self.optimizer.zero_grad()
        vfl_loss.backward()
        self.optimizer.step()

        avg_image_grads = avg_image_embeddings.grad.detach()
        avg_text_grads = avg_text_embeddings.grad.detach()

        parameters_aggregated = [None] * self.config.num_partitions
        
        image_grads_np = avg_image_grads.cpu().numpy()
        text_grads_np = avg_text_grads.cpu().numpy()
        for i in range(len(parameters_aggregated)):
            if i % 2 == 0: 
                parameters_aggregated[i] = image_grads_np
            else: 
                parameters_aggregated[i] = text_grads_np

        parameters_aggregated = ndarrays_to_parameters(parameters_aggregated)

        with torch.no_grad():
            i2t_pred = logits_per_image.argmax(dim=1)
            i2t_acc = (i2t_pred == labels).float().mean().item() * 100
            t2i_pred = logits_per_text.argmax(dim=1)
            t2i_acc = (t2i_pred == labels).float().mean().item() * 100
            avg_acc = (i2t_acc + t2i_acc) / 2

        metrics_aggregated = {
            "vfl/loss": vfl_loss.item(),
            "vfl/loss_img": loss_img.item(),
            "vfl/loss_txt": loss_txt.item(),
            "vfl/i2t_acc": i2t_acc,
            "vfl/t2i_acc": t2i_acc,
            "vfl/avg_acc": avg_acc,
        }

        self.store_results_and_log(
            server_round=rnd,
            tag="aggregate_fit",
            results_dict=metrics_aggregated, 
        )

        return parameters_aggregated, metrics_aggregated

    def get_fit_config_fn(self, server_round):
        """Return a function which returns the fit configuration."""
        def fit_config(server_round):
            return {"server_round": server_round}
        return fit_config

    def configure_evaluate(self, server_round: int, parameters, client_manager):
        """Configure the next round of evaluation."""
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        # Add server_round to the config sent to clients for logging purposes
        config["server_round"] = server_round
        evaluate_ins = fl.common.EvaluateIns(parameters, config)

        # Sample clients
        clients = client_manager.sample(
            num_clients=self.min_evaluate_clients,
            min_num_clients=self.min_evaluate_clients,
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    # Add aggregate_evaluate if needed, but VFL often evaluates implicitly in fit
    # For this setup, VFL accuracy is calculated in aggregate_fit.
    # The client's evaluate is primarily for receiving grads and updating.
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results."""
        if not results:
            logger.log(WARN, f"Round {server_round}: No evaluation results received.")
            return None, {}

        if failures:
            logger.log(WARN, f"Round {server_round}: Evaluation failures: {failures}")

        aggregated_client_metrics = {}
        num_results = len(results)

        for client, evaluate_res in results:
            cid = client.cid
            client_metrics = evaluate_res.metrics

            if not client_metrics:
                logger.log(INFO, f"Round {server_round}: Client {cid} sent empty metrics. Skipping.")
                continue

            for key, value in client_metrics.items():
                if key not in aggregated_client_metrics:
                    aggregated_client_metrics[key] = 0.0
                aggregated_client_metrics[key] += value

        if not aggregated_client_metrics:
            logger.log(INFO, f"Round {server_round}: No metrics to aggregate.")
            return None, {}


        if wandb.run is not None:
            try:
                wandb.log(aggregated_client_metrics, step=server_round)
                logger.log(INFO, f"Round {server_round}: Logged evaluation metrics to WandB.")
            except Exception as e:
                logger.log(WARN, f"Round {server_round}: Failed to log evaluation metrics to WandB: {e}")

        else:
            logger.log(WARN, f"Round {server_round}: WandB is not initialized. Skipping logging.")

        return 0.0, aggregated_client_metrics