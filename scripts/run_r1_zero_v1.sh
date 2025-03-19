ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=1 src/x_r1/grpo.py \
--config recipes/R1_Zero_1dot5B_config_v1.yaml \
> ./output/Qwen2.5-1.5B-R1-GRPO-V1.log 2>&1