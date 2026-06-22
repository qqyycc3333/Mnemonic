
#!/bin/bash
set -e
# source scripts_0/utils.sh
chmod -R 777 ./LLaMA-Factory

# Configurations
model_name=qwen-4b
stage=sft
template=llama3_metamathqa
dataset=metamathqa
train_dataset=metamathqa_train
eval_dataset=gsm8k_eval

# Path
proj_path=./LLaMA-Factory
model_path=./model/qwen-4b
output_dir=
eval_dir=

# Hyperparameters
learning_rate=0.0005
lora_rank=256

# Debug
# cuda_visible_devices=0   
max_samples=50000
# max_new_tokens=128

# create_empty_file ${result_file}
# echo -e "Fine-tuning using {stage}, {dataset}, {model_name}\n" >> ${result_file}

# Training
# llamafactory-cli \
#     train ./config/${dataset}/${model_name}/sft.yaml \
#     dataset=${dataset}_train \
#     output_dir=${output_dir} \
#     max_samples=${max_samples}

sleep 10


# Eval mmlu
llamafactory-cli \
    eval ./config/${dataset}/${model_name}/eval_mmlu.yaml \
    adapter_name_or_path=${output_dir} \
    save_dir=${output_dir}/eval_ckpt_mmlu \
    finetuning_type=lora

sleep 10

# Eval ceval
llamafactory-cli \
    eval ./config/${dataset}/${model_name}/eval_ceval.yaml \
    adapter_name_or_path=${output_dir} \
    save_dir=${output_dir}/eval_ckpt_ceval \
    finetuning_type=lora

sleep 10

# Eval cmmlu
llamafactory-cli \
    eval ./config/${dataset}/${model_name}/eval_cmmlu.yaml \
    adapter_name_or_path=${output_dir} \
    save_dir=${output_dir}/eval_ckpt_cmmlu \
    finetuning_type=lora
