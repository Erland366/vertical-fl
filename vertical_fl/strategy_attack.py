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
from dataclasses import dataclass

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
class ConfigServerAttack:
    lr: float
    num_rounds: int
    batch_size: int

    use_fixed_data: bool

    aggregate_strategy: str

    num_partitions: int | None = None
    project_name: str = "VFL-CLIP-Attack"
    run_name: str = "CLIP-FedAvg"

    log_attack_metrics: bool = True
    attack_model_path: str = None # Path to load the pre-trained attack model
    attack_side_data_size: int = 0 # Number of samples to use for attack evaluation
    whos_attacking: str = "text" # "text" or "image"

class CLIPFederatedStrategyAttack(fl.server.strategy.FedAvg):
    def __init__(self, config: ConfigServerAttack, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.server_model = CLIPServerModel().to(self.device)
        self.config = config
        self.optimizer = torch.optim.AdamW(self.server_model.parameters(), lr=config.lr)

        # TODO: Add config here later on
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
            project=self.config.project_name, 
            name=run_name, 
            config=self.config
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
        """Aggregate fit results."""
        self.config.num_partitions = len(results)
        if self.config.num_partitions % 2 != 0:
             logger.log(WARN, f"Round {rnd}: Uneven number of clients ({self.config.num_partitions}). VFL pairing might be broken.")

        if not self.accept_failures and failures:
            return None, {}

        image_client_results = []
        text_client_results = []
        for client, fit_res in results:
            client_type = fit_res.metrics.get("client-type")
            if client_type == "image":
                params_image = parameters_to_ndarrays(fit_res.parameters)[0]
                if len(params_image) >= 2:
                    image_embeddings_batch = params_image[0]
                    predicted_text_embeddings_batch = params_image[1]
                    image_client_results.append((client.cid, image_embeddings_batch, predicted_text_embeddings_batch))
                else:
                    logger.log(WARN, f"Client {client.cid}: Image client parameters too short ({len(params_image)}). Skipping attack prediction.")
                    image_embeddings_batch = params_image[0]
                    # Add None for missing prediction
                    image_client_results.append((client.cid, image_embeddings_batch, None))

                image_client_results.append((client.cid, image_embeddings_batch))
            elif client_type == "text":
                params_text = parameters_to_ndarrays(fit_res.parameters)
                if len(params_text) >= 2:
                     text_embeddings_batch = params_text[0]
                     predicted_image_embeddings_batch = params_text[1]
                     text_client_results.append((client.cid, text_embeddings_batch, predicted_image_embeddings_batch))
                else:
                     logger.log(WARN, f"Client {client.cid}: Text client parameters too short ({len(params_text)}). Skipping attack prediction.")
                     text_embeddings_batch = params_text[0]
                     text_client_results.append((client.cid, text_embeddings_batch, None)) # Add None for missing prediction
            else:
                logger.log(WARN, f"Client {client.cid}: Unknown client type '{client_type}'. Skipping.")


        if not image_client_results or not text_client_results:
            logger.log(WARN, f"Round {rnd}: Missing image or text client results. Cannot perform VFL or attack evaluation.")
            return None, {"error": "Missing image or text client results"}

        sum_image_embeddings = sum([emb for _, emb in image_client_results])
        sum_text_embeddings = sum([emb for _, emb, _ in text_client_results if _ is not None]) # Sum only from clients that sent text embeddings

        num_image_clients = len(image_client_results)
        num_text_clients = len(text_client_results)
        
        if num_image_clients > 0 and num_text_clients > 0:
             avg_image_embeddings = torch.from_numpy(sum_image_embeddings / num_image_clients).to(self.device)
             avg_text_embeddings = torch.from_numpy(sum_text_embeddings / num_text_clients).to(self.device)
        else:
             logger.log(WARN, f"Round {rnd}: Not enough image or text clients. Skipping VFL step.")
             return None, {"error": "Not enough image or text clients"}

        avg_image_embeddings.requires_grad_(True)
        avg_text_embeddings.requires_grad_(True)

        logits_per_image, logits_per_text = self.server_model(avg_image_embeddings, avg_text_embeddings)

        batch_size = avg_image_embeddings.size(0)
        labels = torch.arange(batch_size, device=self.device).long()

        loss_img = F.cross_entropy(logits_per_text.t(), labels)
        loss_txt = F.cross_entropy(logits_per_text, labels)
        vfl_loss = (loss_img + loss_txt) / 2.0
        logger.log(INFO, f"Round {rnd} VFL loss: {vfl_loss.item()}")

        self.optimizer.zero_grad()
        vfl_loss.backward()
        self.optimizer.step()

        avg_image_grads = avg_image_embeddings.grad.detach()
        avg_text_grads = avg_text_embeddings.grad.detach()

        attack_metrics = {}
        predicted_embs_list_text = [pred_emb for _, _, pred_emb in text_client_results if pred_emb is not None]
        if predicted_embs_list_text:
            sum_predicted_image_embeddings = sum(predicted_embs_list_text)
            avg_predicted_image_embeddings = torch.from_numpy(sum_predicted_image_embeddings / len(predicted_embs_list_text)).to(self.device)

            if avg_predicted_image_embeddings.shape == avg_image_embeddings.shape:
                cosine_sim = F.cosine_similarity(avg_predicted_image_embeddings, avg_image_embeddings, dim=1).mean().item()
                attack_metrics["attack_cosine_similarity_text"] = cosine_sim
                logger.log(INFO, f"Round {rnd} Attack Cosine Similarity: {cosine_sim:.4f}")
            else:
                logger.log(WARN, f"Round {rnd}: Shape mismatch for attack evaluation: Predicted {avg_predicted_image_embeddings.shape}, Actual {avg_image_embeddings.shape}. Skipping attack metric.")

        predicted_embs_list_image = [pred_emb for _, _, pred_emb in image_client_results if pred_emb is not None]
        if predicted_embs_list_image:
            sum_predicted_text_embeddings = sum(predicted_embs_list_image)
            avg_predicted_text_embeddings = torch.from_numpy(sum_predicted_text_embeddings / len(predicted_embs_list_image)).to(self.device)

            if avg_predicted_text_embeddings.shape == avg_text_embeddings.shape:
                cosine_sim = F.cosine_similarity(avg_predicted_text_embeddings, avg_text_embeddings, dim=1).mean().item()
                attack_metrics["attack_cosine_similarity_image"] = cosine_sim
                logger.log(INFO, f"Round {rnd} Attack Cosine Similarity: {cosine_sim:.4f}")
            else:
                logger.log(WARN, f"Round {rnd}: Shape mismatch for attack evaluation: Predicted {avg_predicted_text_embeddings.shape}, Actual {avg_text_embeddings.shape}. Skipping attack metric.")

        parameters_aggregated = [None] * self.config.num_partitions
        for i in range(len(parameters_aggregated)):
            if i % 2 == 0:
                parameters_aggregated[i] = avg_image_grads.cpu().numpy()
            else:
                parameters_aggregated[i] = avg_text_grads.cpu().numpy()
        
        parameters_aggregated = ndarrays_to_parameters(parameters_aggregated)

        with torch.no_grad():
            i2t_pred = logits_per_image.argmax(dim=1)
            i2t_acc = (i2t_pred == labels).float().mean().item() * 100
            
            t2i_pred = logits_per_text.argmax(dim=1)
            t2i_acc = (t2i_pred == labels).float().mean().item() * 100
            
            avg_acc = (i2t_acc + t2i_acc) / 2

        metrics_aggregated = {
            "loss": vfl_loss.item(),
            "loss_img": loss_img.item(),
            "loss_txt": loss_txt.item(),
            "i2t_acc": i2t_acc,
            "t2i_acc": t2i_acc,
            "avg_acc": avg_acc,
            **attack_metrics,
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