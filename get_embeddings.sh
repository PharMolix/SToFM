
export CUDA_VISIBLE_DEVICES=0
python -u get_embedding.py \
--cell_encoder_path ckpt/cell_encoder \
--data_path data/seekgene_sample \
--batch_size 4 \
--config_path ckpt/config.json \
--model_path ckpt/se2transformer.pth \
--output_filename stofm_emb.npy \
--seed 1 
