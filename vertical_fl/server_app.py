from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from vertical_fl.strategy import CLIPFederatedStrategy


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""

    strategy = CLIPFederatedStrategy(
        min_fit_clients=2,
        min_available_clients=2,
        min_evaluate_clients=2,
        accept_failures=False
    )

    num_rounds = context.run_config.get("num-server-rounds", 10)

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Start Flower server
app = ServerApp(server_fn=server_fn)
