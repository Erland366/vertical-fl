from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, logger
from transformers import CLIPProcessor
from vertical_fl.model import CLIPTextClient, CLIPImageClient
from vertical_fl.data_loader import load_fixed_data, load_datasets
from vertical_fl.attack_model import AttackFromTextNetWithGradient, AttackFromImageNetWithGradient
from logging import INFO, WARN
from dataclasses import dataclass
from torch import nn
import wandb
import os
import torch
import functools # NOT IDEAL WTF
import lovely_tensors as lt; lt.monkey_patch()

ATTACK_STATE_DIR = "attack_states"
os.makedirs(ATTACK_STATE_DIR, exist_ok=True)

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
class ConfigClientAttackGradientOnline:
    lr: float
    batch_size: int
    aggregate_strategy: str
    use_fixed_data: bool

    attack_model_path: str | None = None 
    attack_side_data_size: int = 0
    attack_rounds_to_train_online: int = 0
    attack_lr: float = 1e-5
    whos_attacking: str = "text" 
    
    attack_active_online: bool = True 
    log_attack_predictions_client: bool = True 

    # TODO: This is not scalable! We hardcoded the path
    # Ideally, we add the partition id but sadly we cannot pass the path to the evaluation (for saving)
    # We cannot also save the path using class attribute since it always gets instantiated -> Back to None somehow?
    # NOTE: THIS IS NOT WORKING, WE HARDCODED THEM IN THE BOTTOM
    attack_state_image: str = os.path.join(ATTACK_STATE_DIR, f"attack_state_image.pt")
    attack_state_text: str = os.path.join(ATTACK_STATE_DIR, f"attack_state_text.pt")


class TextFlowerClientAttackerGradientOnline(NumPyClient):
    def __init__(self, train_text, eval_image, partition_id: int, config: ConfigClientAttackGradientOnline):
        super().__init__()
        self.properties = {"client_type" : "text"}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = train_text 
        self.eval_image = eval_image
        self.partition_id = partition_id
        self.config = config
        self.model_text = CLIPTextClient().to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", do_rescale=False)
        self.optimizer = torch.optim.AdamW(self.model_text.parameters(), lr=config.lr)

        self.attack_model = None
        self.attack_optimizer = None
        self.attack_state_path = None
        self.loaded_state = False
        # Load victim's encoder (Image Encoder) - frozen
        self.victim_encoder = CLIPImageClient().to(self.device)
        self.victim_encoder.eval()

        for param in self.victim_encoder.parameters():
            param.requires_grad = False

        if self.config.attack_active_online and self.config.whos_attacking == 'text':
            self.attack_state_path = self.config.attack_state_text
            self.attack_model = AttackFromTextNetWithGradient().to(self.device)
            self.loaded_state = False
            if os.path.exists(self.attack_state_path):
                try:
                    self.attack_model.load_state_dict(torch.load(self.attack_state_path, map_location=self.device))
                    logger.log(INFO, f"Text client {partition_id} loaded saved state attack model state from {self.attack_state_path}")
                    self.loaded_state = True
                except Exception as e:
                    logger.log(WARN, f"Text client {partition_id}: Failed to load attack model state from {self.attack_state_path}. Error: {e}")
                    self.loaded_state = False
            elif self.config.attack_model_path and os.path.exists(self.config.attack_model_path):
                try:
                    self.attack_model.load_state_dict(torch.load(self.config.attack_model_path, map_location=self.device))
                    self.loaded_state = True
                    logger.log(INFO, f"Text client {partition_id} loaded attack model from {self.config.attack_model_path}")
                except Exception as e:
                    logger.log(WARN, f"Text client {partition_id}: Failed to load attack model from {self.config.attack_model_path}. Error: {e}")
                    self.loaded_state = False
            else:
                logger.log(WARN, f"Text client {partition_id}: Attack active but model path '{self.config.attack_model_path}' not found or not specified.")

            if self.loaded_state:
                self.attack_optimizer = torch.optim.AdamW(self.attack_model.parameters(), lr = self.config.attack_lr)
            else:
                self.attack_optimizer = None
                logger.log(WARN, f"Text client {partition_id}: Attack model state not loaded. Attack inactive.")
        elif self.config.attack_active_online and self.config.whos_attacking != 'text':
             logger.log(INFO, f"Text client {partition_id}: Attack inactive for this client type.")


    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model_text.state_dict().items()]


    def fit(self, parameters, config):
        self.model_text.eval() 
        text_inputs = self.processor(text=self.data, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        with torch.no_grad():
            text_embeddings = self.model_text(**text_inputs)

        params_to_send = [text_embeddings.cpu().numpy()]
        num_examples = len(self.data)
        metrics = {"client-type": "text"}

        return params_to_send, num_examples, metrics

    def evaluate(self, parameters, config):
        server_round = config.get("server_round", -1)
        client_metrics = {"attack_performed" : False, "online_attack_trained" : False}

        self.model_text.train() 
        self.optimizer.zero_grad()

        text_inputs = self.processor(text=self.data, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        text_embeddings_for_update = self.model_text(**text_inputs)

        try:
            grad_tensor_np = parameters[self.partition_id]
            server_grad = torch.from_numpy(grad_tensor_np).to(self.device)
            if server_grad.shape != text_embeddings_for_update.shape:
                 logger.log(WARN, f"Client {self.partition_id} (Text): Shape mismatch! Grad: {server_grad.shape}, Emb: {text_embeddings_for_update.shape}. Skipping update.")
                 return 0.0, 0, {"error": "Gradient shape mismatch"}

        except IndexError:
            logger.log(WARN, f"Client {self.partition_id} (Text): ERROR - IndexError accessing gradient parameters[{self.partition_id}]")
            return 0.0, 0, {"error": "IndexError getting gradient"}
        except Exception as e:
            logger.log(WARN, f"Client {self.partition_id} (Text): ERROR - Exception getting gradient: {e}")
            return 0.0, 0, {"error": f"Exception getting gradient: {e}"}

        # Online Attack Training Step
        if self.attack_model is not None and self.attack_optimizer is not None and server_round < self.config.attack_rounds_to_train_online:
            logger.log(INFO, "Entering online attack training step for text client.")
            client_metrics["online_attack_trained"] = True
            self.attack_model.train()
            self.attack_optimizer.zero_grad()

            with torch.no_grad():
                image_inputs = self.processor(images=self.eval_image, return_tensors="pt")
                image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
                victim_embedding_truth = self.victim_encoder(**image_inputs)

            predicted_embedding = self.attack_model(text_embeddings_for_update.detach(), server_grad)

            try:
                criterion = nn.CosineEmbeddingLoss()
                cosine_target = torch.ones(predicted_embedding.shape[0], device=self.device)
                attack_loss = criterion(predicted_embedding, victim_embedding_truth.detach(), cosine_target)

                attack_loss.backward()
                self.attack_optimizer.step()
                client_metrics["online_attack_loss"] = attack_loss.item()
                logger.log(INFO, f"Client {self.partition_id} (Text): Attack loss logged: {client_metrics}")

                try:
                    torch.save(self.attack_model.state_dict(), os.path.join(ATTACK_STATE_DIR, f"attack_state_text.pt"))
                    logger.log(INFO, f"Client {self.partition_id} (Text): Attack model state saved to {self.attack_state_path}")
                except Exception as e:
                    logger.log(WARN, f"Client {self.partition_id} (Text): Failed to save attack model state: {e}")

            except Exception as e:
                logger.log(WARN, f"Client {self.partition_id} (Text): Failed to compute attack loss: {e}")
                client_metrics["online_attack_error"] = 1

        try:
            text_embeddings_for_update.backward(server_grad)
            self.optimizer.step()
        except Exception as e:
            logger.log(WARN, f"Client {self.partition_id} (Text): ERROR during model update: {e}")
            client_metrics["vfl_update_error"] = 1



        if self.attack_model is not None:
            client_metrics["attack_performed"] = True
            self.attack_model.eval()
            if text_embeddings_for_update.shape[0] == server_grad.shape[0]: # Check batch size consistency
                attack_input_emb = text_embeddings_for_update.detach()
                attack_input_grad = server_grad

                with torch.no_grad():
                    predicted_image_embedding = self.attack_model(attack_input_emb, attack_input_grad)

                try:
                    client_metrics[f"client_prediction/image_emb_mean"] = predicted_image_embedding.mean().item()
                    client_metrics[f"client_prediction/image_emb_std"] = predicted_image_embedding.std().item()
                    logger.log(INFO, f"Client {self.partition_id} (Text): Attack prediction logged: {client_metrics}")
                except Exception as e:
                    logger.log(WARN, f"Client {self.partition_id} (Text): Failed to log attack predictions to WandB: {e}")
            else:
                 logger.log(WARN, f"Client {self.partition_id} (Text): Batch size mismatch between last embedding ({text_embeddings_for_update.shape[0]}) and gradient ({server_grad.shape[0]}). Skipping attack inference.")


        num_examples = len(self.data)
        # Return dummy loss/metrics as evaluate's primary role here is update/attack
        return 0.0, num_examples, client_metrics


class ImageFlowerClientAttackerGradientOnline(NumPyClient):
    def __init__(self, train_image, eval_text, partition_id: int, config: ConfigClientAttackGradientOnline):
        super().__init__()
        self.properties = {"client_type" : "image"}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = train_image # Batch data for the current round
        self.eval_text = eval_text
        self.partition_id = partition_id
        self.config = config
        self.model_image = CLIPImageClient().to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", do_rescale=False)
        self.optimizer = torch.optim.AdamW(self.model_image.parameters(), lr=config.lr)

        self.attack_model = None
        self.attack_optimizer = None
        self.attack_state_path = None
        self.loaded_state = False
        self.first_step = False

        self.victim_encoder = CLIPTextClient().to(self.device)
        self.victim_encoder.eval()
        for param in self.victim_encoder.parameters():
            param.requires_grad = False

        if self.config.attack_active_online and self.config.whos_attacking == 'image':
            self.attack_state_path = self.config.attack_state_image
            self.attack_model = AttackFromImageNetWithGradient().to(self.device)
            self.loaded_state = False

            if os.path.exists(self.attack_state_path):
                try:
                    self.attack_model.load_state_dict(torch.load(self.attack_state_path, map_location=self.device))
                    logger.log(INFO, f"Image client {partition_id} loaded saved state attack model state from {self.attack_state_path}")
                    self.loaded_state = True
                except Exception as e:
                    logger.log(WARN, f"Image client {partition_id}: Failed to load attack model state from {self.attack_state_path}. Error: {e}")
                    self.loaded_state = False
            elif self.config.attack_model_path and os.path.exists(self.config.attack_model_path):
                try:
                    self.attack_model.load_state_dict(torch.load(self.config.attack_model_path, map_location=self.device))
                    self.loaded_state = True
                    # self.attack_model.eval() # Set to evaluation mode
                    logger.log(INFO, f"Image client {partition_id} loaded attack model from {self.config.attack_model_path}")
                except Exception as e:
                    logger.log(WARN, f"Image client {partition_id}: Failed to load attack model from {self.config.attack_model_path}. Error: {e}")
                    self.loaded_state = False
            else:
                logger.log(WARN, f"Image client {partition_id}: Attack active but model path '{self.config.attack_model_path}' not found or not specified.")

            if self.loaded_state:
                self.attack_optimizer = torch.optim.AdamW(self.attack_model.parameters(), lr = self.config.attack_lr)
            else:
                self.attack_optimizer = None
                logger.log(WARN, f"Image client {partition_id}: Attack model state not loaded. Attack inactive.")
        elif self.config.attack_active_online and self.config.whos_attacking != 'image':
            logger.log(INFO, f"Image client {partition_id}: Attack inactive for this client type.")


    def get_parameters(self, config):
         # Return image model params for potential initial sync
        return [val.cpu().numpy() for _, val in self.model_image.state_dict().items()]


    def fit(self, parameters, config):
        self.model_image.eval()
        image_inputs = self.processor(images=self.data, return_tensors="pt")
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

        with torch.no_grad():
            image_embeddings = self.model_image(**image_inputs)

        params_to_send = [image_embeddings.cpu().numpy()]
        num_examples = len(self.data)
        metrics = {"client-type": "image"}

        return params_to_send, num_examples, metrics

    def evaluate(self, parameters, config):
        server_round = config.get("server_round", -1)
        client_metrics = {
            "attack_performed" : False,
            "online_attack_trained" : False,
        }

        self.model_image.train()
        self.optimizer.zero_grad()

        image_inputs = self.processor(images=self.data, return_tensors="pt")
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
        image_embeddings_for_update = self.model_image(**image_inputs)

        try:
            grad_tensor_np = parameters[self.partition_id]
            server_grad = torch.from_numpy(grad_tensor_np).to(self.device)
            if server_grad.shape != image_embeddings_for_update.shape:
                 logger.log(WARN, f"Client {self.partition_id} (Image): Shape mismatch! Grad: {server_grad.shape}, Emb: {image_embeddings_for_update.shape}. Skipping update.")
                 return 0.0, 0, {"error": "Gradient shape mismatch"}

        except IndexError:
             logger.log(WARN, f"Client {self.partition_id} (Image): ERROR - IndexError accessing gradient parameters[{self.partition_id}]")
             return 0.0, 0, {"error": "IndexError getting gradient"}
        except Exception as e:
             logger.log(WARN, f"Client {self.partition_id} (Image): ERROR - Exception getting gradient: {e}")
             return 0.0, 0, {"error": f"Exception getting gradient: {e}"}

        if self.attack_model is not None and self.attack_optimizer is not None and server_round < self.config.attack_rounds_to_train_online:
            logger.log(INFO, "Entering online attack training step for image client.")
            client_metrics["online_attack_trained"] = True
            self.attack_model.train()
            self.attack_optimizer.zero_grad()

            with torch.no_grad():
                text_inputs = self.processor(text=self.eval_text, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                victim_embedding_truth = self.victim_encoder(**text_inputs)

            predicted_embedding = self.attack_model(image_embeddings_for_update.detach(), server_grad)

            try:
                criterion = nn.CosineEmbeddingLoss()
                cosine_target = torch.ones(predicted_embedding.shape[0], device=self.device)
                attack_loss = criterion(predicted_embedding, victim_embedding_truth.detach(), cosine_target)

                attack_loss.backward()
                self.attack_optimizer.step()
                client_metrics["online_attack_loss"] = attack_loss.item()
                logger.log(INFO, f"Client {self.partition_id} (Image): Attack loss logged: {client_metrics}")

                try:
                    torch.save(self.attack_model.state_dict(), os.path.join(ATTACK_STATE_DIR, f"attack_state_image.pt"))
                    logger.log(INFO, f"Client {self.partition_id} (Image): Attack model state saved to {self.attack_state_path}")
                except Exception as e:
                    logger.log(WARN, f"Client {self.partition_id} (Image): Failed to save attack model state: {e}")
            except Exception as e:
                logger.log(WARN, f"Client {self.partition_id} (Image): Failed to compute attack loss: {e}")
                client_metrics["online_attack_error"] = 1

        try:
            image_embeddings_for_update.backward(server_grad)
            self.optimizer.step()
        except Exception as e:
            logger.log(WARN, f"Client {self.partition_id} (Image): ERROR during model update: {e}")
            client_metrics["vfl_update_error"] = 1


        if self.attack_model is not None:
            client_metrics["attack_performed"] = True
            self.attack_model.eval()
            if image_embeddings_for_update.shape[0] == server_grad.shape[0]: # Check batch size consistency
                attack_input_emb = image_embeddings_for_update.detach()
                attack_input_grad = server_grad

                with torch.no_grad():
                    predicted_text_embedding = self.attack_model(attack_input_emb, attack_input_grad)

                try:
                    client_metrics[f"client_prediction/text_emb_mean"] = predicted_text_embedding.mean().item()
                    client_metrics[f"client_prediction/text_emb_std"] = predicted_text_embedding.std().item()
                    logger.log(INFO, f"Client {self.partition_id} (Image): Attack prediction logged: {client_metrics}")
                except Exception as e:
                    logger.log(WARN, f"Client {self.partition_id} (Image): Failed to log attack predictions to WandB: {e}")
            else:
                 logger.log(WARN, f"Client {self.partition_id} (Image): Batch size mismatch between last embedding ({image_embeddings_for_update.shape[0]}) and gradient ({server_grad.shape[0]}). Skipping attack inference.")

        num_examples = len(self.data)
        return 0.0, num_examples, client_metrics


class TextFlowerClient(NumPyClient):
     def __init__(self, train_text, partition_id: int, config: ConfigClientAttackGradientOnline):
         super().__init__()
         self.properties = {"client_type" : "text"}
         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         self.data = train_text
         self.partition_id = partition_id
         self.config = config 
         self.model_text = CLIPTextClient().to(self.device)
         self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", do_rescale=False)
         self.optimizer = torch.optim.AdamW(self.model_text.parameters(), lr=config.lr)
         logger.log(INFO, f"Text client {partition_id}: Initialized (Non-Attacker).")


     def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model_text.state_dict().items()]

     def fit(self, parameters, config):
         self.model_text.eval()
         text_inputs = self.processor(text=self.data, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
         text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
         with torch.no_grad():
             text_embeddings = self.model_text(**text_inputs)

         params_to_send = [text_embeddings.cpu().numpy()]
         num_examples = len(self.data) 
         metrics = {"client-type": "text"}
         return params_to_send, num_examples, metrics

     def evaluate(self, parameters, config):
         self.model_text.train()
         self.optimizer.zero_grad()
         text_inputs = self.processor(text=self.data, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
         text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
         text_embeddings_for_update = self.model_text(**text_inputs)

         try:
             grad_tensor_np = parameters[self.partition_id]
             server_grad = torch.from_numpy(grad_tensor_np).to(self.device)
             if server_grad.shape != text_embeddings_for_update.shape:
                  logger.log(WARN, f"Client {self.partition_id} (Text/NonAttacker): Shape mismatch! Grad: {server_grad.shape}, Emb: {text_embeddings_for_update.shape}. Skipping update.")
                  return 0.0, 0, {"error": "Gradient shape mismatch"}

             text_embeddings_for_update.backward(server_grad)
             self.optimizer.step()
         except IndexError:
              logger.log(WARN, f"Client {self.partition_id} (Text/NonAttacker): ERROR - IndexError accessing gradient.")
              return 0.0, 0, {"error": "IndexError getting gradient"}
         except Exception as e:
              logger.log(WARN, f"Client {self.partition_id} (Text/NonAttacker): ERROR during update: {e}")
              return 0.0, 0, {"error": f"Exception during update: {e}"}

         num_examples = len(self.data)
         return 0.0, num_examples, {}


class ImageFlowerClient(NumPyClient):
     def __init__(self, train_image, partition_id: int, config: ConfigClientAttackGradientOnline):
         super().__init__()
         self.properties = {"client_type" : "image"}
         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         self.data = train_image
         self.partition_id = partition_id
         self.config = config
         self.model_image = CLIPImageClient().to(self.device)
         self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", do_rescale=False)
         self.optimizer = torch.optim.AdamW(self.model_image.parameters(), lr=config.lr)
         logger.log(INFO, f"Image client {partition_id}: Initialized (Non-Attacker).")


     def get_parameters(self, config):
         return [val.cpu().numpy() for _, val in self.model_image.state_dict().items()]

     def fit(self, parameters, config):
         self.model_image.eval()
         image_inputs = self.processor(images=self.data, return_tensors="pt")
         image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
         with torch.no_grad():
             image_embeddings = self.model_image(**image_inputs)

         params_to_send = [image_embeddings.cpu().numpy()]
         num_examples = len(self.data)
         metrics = {"client-type": "image"}
         return params_to_send, num_examples, metrics

     def evaluate(self, parameters, config):
         self.model_image.train()
         self.optimizer.zero_grad()
         image_inputs = self.processor(images=self.data, return_tensors="pt")
         image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
         image_embeddings_for_update = self.model_image(**image_inputs)

         try:
             grad_tensor_np = parameters[self.partition_id]
             server_grad = torch.from_numpy(grad_tensor_np).to(self.device)
             if server_grad.shape != image_embeddings_for_update.shape:
                  logger.log(WARN, f"Client {self.partition_id} (Image/NonAttacker): Shape mismatch! Grad: {server_grad.shape}, Emb: {image_embeddings_for_update.shape}. Skipping update.")
                  return 0.0, 0, {"error": "Gradient shape mismatch"}

             image_embeddings_for_update.backward(server_grad)
             self.optimizer.step()
         except IndexError:
             logger.log(WARN, f"Client {self.partition_id} (Image/NonAttacker): ERROR - IndexError accessing gradient.")
             return 0.0, 0, {"error": "IndexError getting gradient"}
         except Exception as e:
             logger.log(WARN, f"Client {self.partition_id} (Image/NonAttacker): ERROR during update: {e}")
             return 0.0, 0, {"error": f"Exception during update: {e}"}

         num_examples = len(self.data)
         return 0.0, num_examples, {}

def client_fn(context: Context):
    """Create either an image or text client based on configuration."""
    partition_id = context.node_config.get("partition-id", 0)
    num_partitions = context.node_config.get("num-partitions", 2)
    client_type = "image" if partition_id % 2 == 0 else "text"

    config = ConfigClientAttackGradientOnline(
        lr=context.run_config.get("train.learning-rate", 1e-4),
        batch_size=context.run_config.get("train.batch-size", 16),
        use_fixed_data=context.run_config.get("use-fixed-data", True),
        aggregate_strategy=context.run_config.get("aggregate-strategy", "reduce"),
        attack_model_path=context.run_config.get("attack.model_path", None),
        attack_side_data_size=context.run_config.get("attack.side_data_size", 0),
        whos_attacking=context.run_config.get("attack.whos_attacking", "text"),
        attack_active_online=context.run_config.get("attack.active_online", True),
        log_attack_predictions_client=context.run_config.get("log.log_attack_predictions_client", True),
        attack_rounds_to_train_online=context.run_config.get("attack.rounds_to_train_online", 0),
        attack_lr=context.run_config.get("attack.attack_lr", 1e-5),
    )

    logger.log(INFO, f"Client {partition_id} initializing. Type: {client_type}, Attacker: {config.whos_attacking == client_type and config.attack_active_online}")
    global _CACHED_IMAGE_LOADER, _CACHED_TEXT_LOADER
    global _CACHED_IMAGE_DATA, _CACHED_TEXT_DATA

    if config.use_fixed_data:
        global _CACHED_IMAGE_DATA, _CACHED_TEXT_DATA
        
        if _CACHED_IMAGE_DATA is None or _CACHED_TEXT_DATA is None:
            _CACHED_IMAGE_DATA, _CACHED_TEXT_DATA = get_fixed_data()
        if client_type == "image":
            if config.whos_attacking == "image":
                return ImageFlowerClientAttackerGradientOnline(_CACHED_IMAGE_DATA, _CACHED_TEXT_DATA, partition_id, config).to_client()
            else:
                return ImageFlowerClient(_CACHED_IMAGE_DATA, partition_id, config).to_client()
        elif client_type == "text":
            if config.whos_attacking == "text":
                return TextFlowerClientAttackerGradientOnline(_CACHED_TEXT_DATA, _CACHED_IMAGE_DATA, partition_id, config).to_client()
            else:
                return TextFlowerClient(_CACHED_TEXT_DATA, partition_id, config).to_client()
    else:

        
        if _CACHED_IMAGE_LOADER is None or _CACHED_TEXT_LOADER is None:
            _CACHED_IMAGE_LOADER, _CACHED_TEXT_LOADER = get_datasets(config.batch_size)

        train_image_iterator = _CACHED_IMAGE_LOADER
        train_text_iterator = _CACHED_TEXT_LOADER

        train_image = next(train_image_iterator)
        train_text = next(train_text_iterator)
        _CACHED_IMAGE_DATA = train_image
        _CACHED_TEXT_DATA = train_text

        if client_type == "image":
            if config.whos_attacking == "image":
                return ImageFlowerClientAttackerGradientOnline(_CACHED_IMAGE_DATA, _CACHED_TEXT_DATA, partition_id, config).to_client()
            else:
                return ImageFlowerClient(_CACHED_IMAGE_DATA, partition_id, config).to_client()
        elif client_type == "text":
            if config.whos_attacking == "text":
                return TextFlowerClientAttackerGradientOnline(_CACHED_TEXT_DATA, _CACHED_IMAGE_DATA, partition_id, config).to_client()
            else:
                return TextFlowerClient(_CACHED_TEXT_DATA, partition_id, config).to_client()

    raise ValueError(f"Unknown client type: {client_type}")

app = ClientApp(
    client_fn=client_fn,
)