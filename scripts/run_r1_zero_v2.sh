ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=1 src/grpo.py \
--config recipes/R1_Zero_0dot5B_config_v1.yaml \
> ./output/Qwen2.5-0.5B-R1-GRPO-V1.log 2>&1