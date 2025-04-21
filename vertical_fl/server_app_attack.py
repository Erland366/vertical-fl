from flwr.common import Context, logger
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from dataclasses import asdict
from logging import INFO, WARN

from vertical_fl.strategy_attack import ConfigServerAttack, CLIPFederatedStrategyAttack


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""
    run_conf = context.run_config
    config = ConfigServerAttack(
        lr=run_conf.get("train.learning-rate", 5e-5),
        num_rounds=run_conf.get("num-server-rounds", 100),
        batch_size=run_conf.get("train.batch-size", 16),
        use_fixed_data=run_conf.get("use-fixed-data", False),
        aggregate_strategy=run_conf.get("aggregate-strategy", "reduce"),
        project_name=run_conf.get("log.project_name", "VFL-CLIP-Attack"),
        run_name=run_conf.get("log.run_name", "CLIP-Attack-Exp"),
        log_attack_metrics=run_conf.get("log.log-attack-metrics", False),
        attack_model_path=run_conf.get("attack.model_path", None),
        attack_side_data_size=run_conf.get("attack.side_data_size", 0),
        whos_attacking=run_conf.get("attack.whos_attacking", "text"),  # "text" or "image"
    )

    logger.log(INFO, f"Server initializing with config: {config}")

    num_expected_clients = run_conf.get("num-supernodes", 2)
    min_clients = max(2, num_expected_clients)

    strategy = CLIPFederatedStrategyAttack(
        config=config,
        min_fit_clients=min_clients,         
        min_available_clients=min_clients,   
        min_evaluate_clients=min_clients,    
        accept_failures=True 
    )

    server_config = ServerConfig(num_rounds=config.num_rounds)

    return ServerAppComponents(strategy=strategy, config=server_config)

app = ServerApp(
    server_fn=server_fn,
)
