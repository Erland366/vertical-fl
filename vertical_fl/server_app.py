from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from vertical_fl.strategy import ConfigServer, CLIPFederatedStrategy


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""
    call_config = context.run_config.get
    config = ConfigServer(
        lr=call_config("train.learning-rate", 1e-4),
        num_rounds=call_config("num-server-rounds", 10),
        batch_size=call_config("train.batch-size", 128),
        use_fixed_data=call_config("use-fixed-data", True),
        run_name=call_config("log.run_name", "CLIP-FedAvg"),
        project_name=call_config("log.project_name", "VFL-CLIP"),
        aggregate_strategy=call_config("aggregate-strategy", "reduce"),
    )

    strategy = CLIPFederatedStrategy(
        config=config,
        min_fit_clients=2,
        min_available_clients=2,
        min_evaluate_clients=2,
        accept_failures=False
    )

    config = ServerConfig(num_rounds=config.run_name)

    return ServerAppComponents(strategy=strategy, config=config)

# Start Flower server
app = ServerApp(server_fn=server_fn)
