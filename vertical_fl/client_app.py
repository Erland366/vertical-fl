from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import torch
from torch.utils.data import DataLoader

from vertical_fl.task import load_data
from vertical_fl.model import Qwen2ForCausalLMWorker
from vertical_fl.dataset import get_tokenizer_and_data_collator, load_data, _group_texts

from transformers import AutoConfig

import os
import warnings

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

class FlowerClient(NumPyClient):
    def __init__(
        self, 
        v_split_id: int, 
        lr: float,
        train_set,
        tokenizer,
        data_collator,
        num_rounds: int,
        block_size: int = 512
    ):
        self.v_split_id = v_split_id
        self.train_set = train_set
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.data_collator = data_collator 

        self.config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
        self.config.tie_word_embeddings = False
        self.config.dtype = "bfloat16"

        self.model_worker = Qwen2ForCausalLMWorker(self.config)
        self.optimizer = torch.optim.AdamW(
            self.model_worker.parameters(),
            lr=lr or 1e-4,
            betas=(0.9, 0.999)
        )
        self.train_loader = self.get_dataloader()
        self.data_iterator = iter(self.train_loader)

    @property
    def data(self):
        """Fetch a new batch of data from the dataloader each time this is accessed."""
        try:
            batch = next(self.data_iterator)
        except (StopIteration, AttributeError):
            self.data_iterator = iter(self.train_loader)
            batch = next(self.data_iterator)
        
        return batch

    def get_dataloader(self):
        tokenized_dataset = self.train_set.map(lambda x: self.tokenizer(x["text"], return_tensors="pt", padding="max_length", truncation=True), num_proc=2, remove_columns=["text"]).to(self.device)
        lm_datasets = tokenized_dataset.map(
            lambda x: _group_texts(x, self.block_size),
            num_proc=2
        )

        train_dataloader = DataLoader(
            lm_datasets,
            shuffle=True,
            collate_fn=self.default_data_collator,
            batch_size=2
        )

        return train_dataloader

    def get_parameters(self, config):
        pass

    def fit(self, parameters, config):
        embedding = self.model_worker(**self.data)

        return [embedding.detach().numpy()], 1, {}

    def evaluate(self, parameters, config):
        self.model_worker.zero_grad()
        embedding = self.model_worker(self.data)
        embedding.backward(torch.from_numpy(parameters[int(self.v_split_id)]))
        self.optimizer.step()
        return 0.0, 1, {}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    client_trainset = load_data(partition_id, num_partitions=num_partitions)
    lr = context.run_config["learning-rate"]
    return FlowerClient(v_split_id, partition, lr).to_client()


app = ClientApp(
    client_fn=client_fn,
)
