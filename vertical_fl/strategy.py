import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from vertical_fl.model import Qwen2ForCausalLMServer
from transformers import AutoConfig


class Strategy(fl.server.strategy.FedAvg):
    def __init__(self, labels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
        self.config.tie_word_embeddings = False
        self.config.dtype = "bfloat16"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_server = Qwen2ForCausalLMServer(self.config).to(self.device)
        self.initial_parameters = ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in self.model_server.state_dict().items()]
        )
        self.optimizer = optim.AdamW(
            self.model_server.parameters(), 
            lr=1e-4, 
            betas=(0.9, 0.999)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.labels = labels # [B, S]

    def aggregate_fit(
        self,
        rnd,
        results,
        failures,
    ):
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        embedding_results = [
            torch.from_numpy(parameters_to_ndarrays(fit_res.parameters)[0])
            for _, fit_res in results
        ]

        # The parameters above doesn't necesarily means parameter, in this case
        # I think it means the activation

        # Concat on the hidden dimension
        embeddings_aggregated = torch.cat(embedding_results, dim=-1)

        # This should be already the hidden state
        embedding_server = embeddings_aggregated.detach().requires_grad_()
        outputs = self.model_server(input_embeds=embedding_server, labels=self.labels)
        loss = outputs.loss
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        # Assuming we have 3 nodes, that means we split the model into 4 parts on the hidden dim
        # (since that's the amount that we can do for splitting equally)
        grads = embedding_server.grad.split([int(self.config.hidden_size // 4)] * 4, dim=-1)
        np_grads = [grad.numpy() for grad in grads]
        parameters_aggregated = ndarrays_to_parameters(np_grads)

        with torch.no_grad():
            outputs = self.model_server(input_embeds=embedding_server, labels=self.labels)
            loss = outputs.loss

        metrics_aggregated = {"loss": loss}

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        rnd,
        results,
        failures,
    ):
        return None, {}
