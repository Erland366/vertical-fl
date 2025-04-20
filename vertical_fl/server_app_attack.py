from flwr.common import Context, logger
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from dataclasses import asdict # Import asdict
from logging import INFO, WARN

# Import strategy and config from your project structure
from vertical_fl.strategy_attack import ConfigServerAttack, CLIPFederatedStrategyAttack


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""
    # Load configuration from the run config provided by flwr run
    run_conf = context.run_config
    config = ConfigServerAttack(
        lr=run_conf.get("train.learning-rate", 5e-5),
        num_rounds=run_conf.get("num-server-rounds", 100),
        batch_size=run_conf.get("train.batch-size", 16),
        use_fixed_data=run_conf.get("use-fixed-data", False),
        aggregate_strategy=run_conf.get("aggregate-strategy", "reduce"),
        # Logging config
        project_name=run_conf.get("log.project_name", "VFL-CLIP-Attack"),
        run_name=run_conf.get("log.run_name", "CLIP-Attack-Exp"),
        log_attack_metrics=run_conf.get("log.log-attack-metrics", False),
        attack_model_path=run_conf.get("attack.model_path", None),
        attack_side_data_size=run_conf.get("attack.side_data_size", 0),
    )

    logger.log(INFO, f"Server initializing with config: {config}")


    # Instantiate the custom strategy with the loaded configuration
    # Determine min clients based on expected pairs (e.g., need at least 1 image + 1 text)
    # This might need adjustment based on how clients join/fail.
    # For simulation, we often expect all clients defined in pyproject.toml.
    num_expected_clients = run_conf.get("num-supernodes", 2) # Get from federation config if possible
    min_clients = max(2, num_expected_clients) # Ensure at least one pair

    strategy = CLIPFederatedStrategyAttack(
        config=config,
        min_fit_clients=min_clients,         # Minimum clients for training round
        min_available_clients=min_clients,   # Minimum clients required overall
        min_evaluate_clients=min_clients,    # Minimum clients for evaluation round
        accept_failures=True # Allow rounds to proceed even if some clients fail
        # Pass other FedAvg parameters if needed (e.g., fit_metrics_aggregation_fn)
    )

    # Configure the Flower server
    server_config = ServerConfig(num_rounds=config.num_rounds)

    # Return the components for the ServerApp
    return ServerAppComponents(strategy=strategy, config=server_config)

app = ServerApp(
    server_fn=server_fn,
)
