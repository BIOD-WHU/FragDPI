CUDA_VISIBLE_DEVICES=1 python run_interaction.py \
--task=test_mol \
--b=64 \
--output=./predict/test \
--config=./config/config_layer_6_mol.json \
--init=./path/to/your/model 