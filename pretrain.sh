CUDA_VISIBLE_DEVICES=7 python run_pretraining.py \
--task=pre-train \
--epochs=30 \
--lr=1e-4 \
--savedir=train \
--config=./config/config_layer_6_mol.json 