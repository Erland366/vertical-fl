from itertools import chain

from transformers import default_data_collator, AutoTokenizer

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

FDS = None

def get_tokenizer_and_data_collator(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding="right")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, default_data_collator

def load_data(partition_id: int, num_partitions: int, dataset_name: str="Erland/fineweb-edu-cleaned-simplified-subset-with-eval"):
    global FDS
    if FDS is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        FDS = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train" : partitioner}
        )
    client_trainset = FDS.load_partition(partition_id=partition_id, split="train")
    
    return client_trainset


def _group_texts(examples, block_size):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result