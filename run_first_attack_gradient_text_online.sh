# # no online training

# WANDB_MODE="online" flwr run . --run-config 'attack.whos_attacking="image" attack.rounds_to_train_online=0 attack.model_path="attack_models_image_gradient/attack_model_sidesize_10_loss_cosine.pth" attack.side_data_size=10'
# WANDB_MODE="online" flwr run . --run-config 'attack.whos_attacking="image" attack.rounds_to_train_online=0 attack.model_path="attack_models_image_gradient/attack_model_sidesize_100_loss_cosine.pth" attack.side_data_size=100'
# WANDB_MODE="online" flwr run . --run-config 'attack.whos_attacking="image" attack.rounds_to_train_online=0 attack.model_path="attack_models_image_gradient/attack_model_sidesize_250_loss_cosine.pth" attack.side_data_size=250'
# WANDB_MODE="online" flwr run . --run-config 'attack.whos_attacking="image" attack.rounds_to_train_online=0 attack.model_path="attack_models_image_gradient/attack_model_sidesize_500_loss_cosine.pth" attack.side_data_size=500'
# WANDB_MODE="online" flwr run . --run-config 'attack.whos_attacking="image" attack.rounds_to_train_online=0 attack.model_path="attack_models_image_gradient/attack_model_sidesize_1000_loss_cosine.pth" attack.side_data_size=1000'

# 10 online training

WANDB_MODE="online" flwr run . --run-config 'attack.whos_attacking="image" attack.rounds_to_train_online=10 attack.model_path="attack_models_image_gradient/attack_model_sidesize_10_loss_cosine.pth" attack.side_data_size=10'
WANDB_MODE="online" flwr run . --run-config 'attack.whos_attacking="image" attack.rounds_to_train_online=10 attack.model_path="attack_models_image_gradient/attack_model_sidesize_100_loss_cosine.pth" attack.side_data_size=100'
WANDB_MODE="online" flwr run . --run-config 'attack.whos_attacking="image" attack.rounds_to_train_online=10 attack.model_path="attack_models_image_gradient/attack_model_sidesize_250_loss_cosine.pth" attack.side_data_size=250'
WANDB_MODE="online" flwr run . --run-config 'attack.whos_attacking="image" attack.rounds_to_train_online=10 attack.model_path="attack_models_image_gradient/attack_model_sidesize_500_loss_cosine.pth" attack.side_data_size=500'
WANDB_MODE="online" flwr run . --run-config 'attack.whos_attacking="image" attack.rounds_to_train_online=10 attack.model_path="attack_models_image_gradient/attack_model_sidesize_1000_loss_cosine.pth" attack.side_data_size=1000'

# 50 online training

WANDB_MODE="online" flwr run . --run-config 'attack.whos_attacking="image" attack.rounds_to_train_online=50 attack.model_path="attack_models_image_gradient/attack_model_sidesize_10_loss_cosine.pth" attack.side_data_size=10'
WANDB_MODE="online" flwr run . --run-config 'attack.whos_attacking="image" attack.rounds_to_train_online=50 attack.model_path="attack_models_image_gradient/attack_model_sidesize_100_loss_cosine.pth" attack.side_data_size=100'
WANDB_MODE="online" flwr run . --run-config 'attack.whos_attacking="image" attack.rounds_to_train_online=50 attack.model_path="attack_models_image_gradient/attack_model_sidesize_250_loss_cosine.pth" attack.side_data_size=250'
WANDB_MODE="online" flwr run . --run-config 'attack.whos_attacking="image" attack.rounds_to_train_online=50 attack.model_path="attack_models_image_gradient/attack_model_sidesize_500_loss_cosine.pth" attack.side_data_size=500'
WANDB_MODE="online" flwr run . --run-config 'attack.whos_attacking="image" attack.rounds_to_train_online=50 attack.model_path="attack_models_image_gradient/attack_model_sidesize_1000_loss_cosine.pth" attack.side_data_size=1000'

# 99 online training

WANDB_MODE="online" flwr run . --run-config 'attack.whos_attacking="image" attack.rounds_to_train_online=99 attack.model_path="attack_models_image_gradient/attack_model_sidesize_10_loss_cosine.pth" attack.side_data_size=10'
WANDB_MODE="online" flwr run . --run-config 'attack.whos_attacking="image" attack.rounds_to_train_online=99 attack.model_path="attack_models_image_gradient/attack_model_sidesize_100_loss_cosine.pth" attack.side_data_size=100'
WANDB_MODE="online" flwr run . --run-config 'attack.whos_attacking="image" attack.rounds_to_train_online=99 attack.model_path="attack_models_image_gradient/attack_model_sidesize_250_loss_cosine.pth" attack.side_data_size=250'
WANDB_MODE="online" flwr run . --run-config 'attack.whos_attacking="image" attack.rounds_to_train_online=99 attack.model_path="attack_models_image_gradient/attack_model_sidesize_500_loss_cosine.pth" attack.side_data_size=500'
WANDB_MODE="online" flwr run . --run-config 'attack.whos_attacking="image" attack.rounds_to_train_online=99 attack.model_path="attack_models_image_gradient/attack_model_sidesize_1000_loss_cosine.pth" attack.side_data_size=1000'