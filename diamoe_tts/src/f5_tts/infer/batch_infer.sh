CUDA_VISIBLE_DEVICES=0 python ./batch_infer.py \
--model test \
--use_ema false \
--use_moe true \
--expert_type MLP \
--num_exps 9 \
--moe_topK 1 \
--gen_file path/to/testset.txt \
--ckpt_file path/to/ckpt.pt \
--vocab_file diamoettsv1/diamoe_tts/data/vocab.txt  \
--ref_audio path/to/reference.wav \
--ref_text path/to/reference.txt \
--output_dir path/to/output_dir












