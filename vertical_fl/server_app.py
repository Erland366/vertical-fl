from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from vertical_fl.strategy import CLIPFederatedStrategy


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""
    lr = context.run_config.get("train.learning-rate", 1e-4)
    num_rounds = context.run_config.get("num-server-rounds", 10)

    strategy = CLIPFederatedStrategy(
        lr=lr,
        min_fit_clients=2,
        min_available_clients=2,
        min_evaluate_clients=2,
        accept_failures=False
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Start Flower server
app = ServerApp(server_fn=server_fn)
