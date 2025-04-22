from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, logger
from transformers import CLIPProcessor
from vertical_fl.model import CLIPTextClient, CLIPImageClient
from vertical_fl.data_loader import load_fixed_data, load_datasets
from vertical_fl.attack_model import AttackFromTextNetWithGradient, AttackFromImageNetWithGradient
from logging import INFO
from dataclasses import dataclass
import wandb
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
class ConfigClientAttackGradient:
    lr: float
    batch_size: int
    aggregate_strategy: str
    use_fixed_data: bool

    attack_model_path: str | None = None 
    attack_side_data_size: int = 0 
    whos_attacking: str = "text" 

    
    attack_active_online: bool = True 
    log_attack_predictions_client: bool = True 


class TextFlowerClientAttackerGradient(NumPyClient):
    def __init__(self, train_text, partition_id: int, config: ConfigClientAttackGradient):
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
        self.last_fit_embedding = None 

        if self.config.attack_active_online and self.config.whos_attacking == 'text':
            if self.config.attack_model_path and os.path.exists(self.config.attack_model_path):
                try:
                    self.attack_model = AttackFromTextNetWithGradient().to(self.device)
                    self.attack_model.load_state_dict(torch.load(self.config.attack_model_path, map_location=self.device))
                    self.attack_model.eval() 
                    logger.log(INFO, f"Text client {partition_id} loaded attack model from {self.config.attack_model_path}")
                except Exception as e:
                    logger.log(WARN, f"Text client {partition_id}: Failed to load attack model from {self.config.attack_model_path}. Error: {e}")
                    self.attack_model = None
            else:
                logger.log(WARN, f"Text client {partition_id}: Attack active but model path '{self.config.attack_model_path}' not found or not specified.")
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

        self.last_fit_embedding = text_embeddings.detach().clone()

        params_to_send = [text_embeddings.cpu().numpy()]
        num_examples = len(self.data)
        metrics = {"client-type": "text"}

        return params_to_send, num_examples, metrics

    def evaluate(self, parameters, config):
        server_round = config.get("server_round", -1) 

        self.model_text.train() 
        self.optimizer.zero_grad()

        text_inputs = self.processor(text=self.data, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        text_embeddings_for_update = self.model_text(**text_inputs)

        # 2. Get Gradient from Server
        try:
            # Parameters now contain the gradients, indexed by partition ID (or a better mapping)
            # Assuming simple modulo mapping for simulation
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


        # 3. Perform Attack Inference (if active attacker)
        if self.attack_model is not None and self.last_fit_embedding is not None:
            if self.last_fit_embedding.shape[0] == server_grad.shape[0]: # Check batch size consistency
                attack_input_emb = self.last_fit_embedding
                attack_input_grad = server_grad

                with torch.no_grad(): # Inference only
                    predicted_image_embedding = self.attack_model(attack_input_emb, attack_input_grad)

                # Log predictions using wandb
                if self.config.log_attack_predictions_client and wandb.run is not None:
                    try:
                        log_data = {
                             f"client_{self.partition_id}_prediction/image_emb_mean": predicted_image_embedding.mean().item(),
                             f"client_{self.partition_id}_prediction/image_emb_std": predicted_image_embedding.std().item(),
                             # "client_{self.partition_id}_predicted_image_embedding": wandb.Histogram(predicted_image_embedding.cpu().numpy()),
                        }
                        wandb.log(log_data, step=server_round)
                    except Exception as e:
                         logger.log(WARN, f"Client {self.partition_id} (Text): Failed to log attack predictions to WandB: {e}")
            else:
                 logger.log(WARN, f"Client {self.partition_id} (Text): Batch size mismatch between last embedding ({self.last_fit_embedding.shape[0]}) and gradient ({server_grad.shape[0]}). Skipping attack inference.")


        # 4. Perform VFL Model Update
        try:
            text_embeddings_for_update.backward(server_grad)
            self.optimizer.step()
        except Exception as e:
             logger.log(WARN, f"Client {self.partition_id} (Text): ERROR during model update: {e}")
             # Don't return error here, just log, as attack might have succeeded

        num_examples = len(self.data)
        # Return dummy loss/metrics as evaluate's primary role here is update/attack
        return 0.0, num_examples, {"attack_performed": (self.attack_model is not None)}


class ImageFlowerClientAttackerGradient(NumPyClient):
    def __init__(self, train_image, partition_id: int, config: ConfigClientAttackGradient):
        super().__init__()
        self.properties = {"client_type" : "image"}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = train_image # Batch data for the current round
        self.partition_id = partition_id
        self.config = config
        self.model_image = CLIPImageClient().to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", do_rescale=False)
        self.optimizer = torch.optim.AdamW(self.model_image.parameters(), lr=config.lr)

        self.attack_model = None
        self.last_fit_embedding = None # Store embedding from fit

        if self.config.attack_active_online and self.config.whos_attacking == 'image':
            if self.config.attack_model_path and os.path.exists(self.config.attack_model_path):
                try:
                     # Ensure correct attack model is loaded
                    self.attack_model = AttackFromImageNetWithGradient().to(self.device)
                    self.attack_model.load_state_dict(torch.load(self.config.attack_model_path, map_location=self.device))
                    self.attack_model.eval() # Set to evaluation mode
                    logger.log(INFO, f"Image client {partition_id} loaded attack model from {self.config.attack_model_path}")
                except Exception as e:
                    logger.log(WARN, f"Image client {partition_id}: Failed to load attack model from {self.config.attack_model_path}. Error: {e}")
                    self.attack_model = None
            else:
                logger.log(WARN, f"Image client {partition_id}: Attack active but model path '{self.config.attack_model_path}' not found or not specified.")
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

        self.last_fit_embedding = image_embeddings.detach().clone()

        params_to_send = [image_embeddings.cpu().numpy()]
        num_examples = len(self.data)
        metrics = {"client-type": "image"}

        return params_to_send, num_examples, metrics

    def evaluate(self, parameters, config):
        server_round = config.get("server_round", -1)

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

        if self.attack_model is not None and self.last_fit_embedding is not None:
            if self.last_fit_embedding.shape[0] == server_grad.shape[0]: # Check batch size consistency
                attack_input_emb = self.last_fit_embedding
                attack_input_grad = server_grad

                with torch.no_grad():
                    predicted_text_embedding = self.attack_model(attack_input_emb, attack_input_grad)

                if self.config.log_attack_predictions_client and wandb.run is not None:
                    try:
                        log_data = {
                            f"client_{self.partition_id}_prediction/text_emb_mean": predicted_text_embedding.mean().item(),
                            f"client_{self.partition_id}_prediction/text_emb_std": predicted_text_embedding.std().item(),
                            # "client_{self.partition_id}_predicted_text_embedding": wandb.Histogram(predicted_text_embedding.cpu().numpy()),
                        }
                        wandb.log(log_data, step=server_round)
                    except Exception as e:
                        logger.log(WARN, f"Client {self.partition_id} (Image): Failed to log attack predictions to WandB: {e}")
            else:
                 logger.log(WARN, f"Client {self.partition_id} (Image): Batch size mismatch between last embedding ({self.last_fit_embedding.shape[0]}) and gradient ({server_grad.shape[0]}). Skipping attack inference.")


        try:
            image_embeddings_for_update.backward(server_grad)
            self.optimizer.step()
        except Exception as e:
             logger.log(WARN, f"Client {self.partition_id} (Image): ERROR during model update: {e}")

        num_examples = len(self.data)
        return 0.0, num_examples, {"attack_performed": (self.attack_model is not None)}


class TextFlowerClient(NumPyClient):
     def __init__(self, train_text, partition_id: int, config: ConfigClientAttackGradient):
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

     def __init__(self, train_image, partition_id: int, config: ConfigClientAttackGradient):
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

    config = ConfigClientAttackGradient(
        lr=context.run_config.get("train.learning-rate", 1e-4),
        batch_size=context.run_config.get("train.batch-size", 16),
        use_fixed_data=context.run_config.get("use-fixed-data", True),
        aggregate_strategy=context.run_config.get("aggregate-strategy", "reduce"),
        attack_model_path=context.run_config.get("attack.model_path", None),
        attack_side_data_size=context.run_config.get("attack.side_data_size", 0),
        whos_attacking=context.run_config.get("attack.whos_attacking", "text"),
        attack_active_online=context.run_config.get("attack.active_online", True),
        log_attack_predictions_client=context.run_config.get("log.log_attack_predictions_client", True),
    )

    logger.log(INFO, f"Client {partition_id} initializing. Type: {client_type}, Attacker: {config.whos_attacking == client_type and config.attack_active_online}")


    if config.use_fixed_data:
        global _CACHED_IMAGE_DATA, _CACHED_TEXT_DATA
        
        if _CACHED_IMAGE_DATA is None or _CACHED_TEXT_DATA is None:
            _CACHED_IMAGE_DATA, _CACHED_TEXT_DATA = get_fixed_data()
        if client_type == "image":
            if config.whos_attacking == "image":
                return ImageFlowerClientAttackerGradient(_CACHED_IMAGE_DATA, partition_id, config).to_client()
            else:
                return ImageFlowerClient(_CACHED_IMAGE_DATA, partition_id, config).to_client()
        elif client_type == "text":
            if config.whos_attacking == "text":
                return TextFlowerClientAttackerGradient(_CACHED_TEXT_DATA, partition_id, config).to_client()
            else:
                return TextFlowerClient(_CACHED_TEXT_DATA, partition_id, config).to_client()
    else:
        global _CACHED_IMAGE_LOADER, _CACHED_TEXT_LOADER
        
        if _CACHED_IMAGE_LOADER is None or _CACHED_TEXT_LOADER is None:
            _CACHED_IMAGE_LOADER, _CACHED_TEXT_LOADER = get_datasets(config.batch_size)

        train_image_iterator = _CACHED_IMAGE_LOADER
        train_text_iterator = _CACHED_TEXT_LOADER
        if client_type == "image":
            train_image = next(train_image_iterator)
            logger.log(INFO, f"{train_image}")
            if config.whos_attacking == "image":
                return ImageFlowerClientAttackerGradient(train_image, partition_id, config).to_client()
            else:
                return ImageFlowerClient(train_image, partition_id, config).to_client()
        elif client_type == "text":
            train_text = next(train_text_iterator)
            if config.whos_attacking == "text":
                return TextFlowerClientAttackerGradient(train_text, partition_id, config).to_client()
            else:
                return TextFlowerClient(train_text, partition_id, config).to_client()

    raise ValueError(f"Unknown client type: {client_type}")

app = ClientApp(
    client_fn=client_fn,
)