CUDA_VISIBLE_DEVICES=7 python run_interaction.py \
--b=64 --task=train_mol --epochs=30 --lr=1e-5 \
--savedir=lr-1e-5-batch-64-e-30-layer6-1125 \
--config=./config/config_layer_6_mol.json \
--init=./path/to/your/pretrain/model
