echo "Running offline attack model training with different side data sizes and loss types on image..."
python vertical_fl/train_attack_model_offline.py --attacker_type image --side_data_size 10 --epochs 10 --loss_type cosine
python vertical_fl/train_attack_model_offline.py --attacker_type image --side_data_size 100 --epochs 10 --loss_type cosine
python vertical_fl/train_attack_model_offline.py --attacker_type image --side_data_size 250 --epochs 10 --loss_type cosine
python vertical_fl/train_attack_model_offline.py --attacker_type image --side_data_size 500 --epochs 10 --loss_type cosine
python vertical_fl/train_attack_model_offline.py --attacker_type image --side_data_size 1000 --epochs 10 --loss_type cosine


echo "Running offline attack model training with different side data sizes and loss types on text..."
python vertical_fl/train_attack_model_offline.py --attacker_type text --side_data_size 10 --epochs 10 --loss_type cosine
python vertical_fl/train_attack_model_offline.py --attacker_type text --side_data_size 100 --epochs 10 --loss_type cosine
python vertical_fl/train_attack_model_offline.py --attacker_type text --side_data_size 250 --epochs 10 --loss_type cosine
python vertical_fl/train_attack_model_offline.py --attacker_type text --side_data_size 500 --epochs 10 --loss_type cosine
python vertical_fl/train_attack_model_offline.py --attacker_type text --side_data_size 1000 --epochs 10 --loss_type cosine