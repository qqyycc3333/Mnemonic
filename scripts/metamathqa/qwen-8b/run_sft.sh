
#!/bin/bash
# set -e
# source scripts_0/utils.sh
chmod -R 777 ./LLaMA-Factory

# Configurations
stage=sft

model_name=qwen-8b
template=llama3_metamathqa

dataset=metamathqa
train_dataset=metamathqa_train
# eval_dataset=gsm8k_eval

# Path
# proj_path=./LLaMA-Factory
model_path=../model/qwen-8b
your_model_path="./model/qwen-8b"
output_dir=
output_path=
# Hyperparameters
learning_rate=0.0005
lora_rank=256

# Debug
# cuda_visible_devices=0   
max_samples=50000
# max_new_tokens=128

# create_empty_file ${result_file}
# echo -e "Fine-tuning using {stage}, {dataset}, {model_name}\n" >> ${result_file}

# for loraplus_lr_ratio in 2 4 6
# do
# output_dir=${output_path}_${loraplus_lr_ratio}

cd LLaMA-Factory
# Training
llamafactory-cli \
    train ../config/${dataset}/${model_name}/sft.yaml \
    dataset=${dataset}_train \
    output_dir=${output_dir} \
    max_samples=${max_samples}

cd ..

# # Eval
# accelerate launch -m lm_eval \
#     --model hf \
#     --model_args pretrained=${your_model_path},peft=${output_dir},trust_remote_code=True \
#     --gen_kwargs max_new_tokens=128 \
#     --tasks gsm8k \
#     --batch_size 8 \
#     --num_fewshot 0 \
#     --output_path ${output_dir}/eval_sft_gsm8k \
#     --seed 42 \

# accelerate launch -m lm_eval \
#         --model hf \
#         --model_args pretrained=${your_model_path},peft=${output_dir},trust_remote_code=True \
#         --gen_kwargs max_new_tokens=3 \
#         --tasks arc_challenge \
#         --batch_size 8 \
#         --num_fewshot 25 \
#         --output_path ${output_dir}/eval_sft_arc_challenge \
#         --seed 42

# accelerate launch -m lm_eval \
#     --model hf \
#     --model_args pretrained=${your_model_path},peft=${output_dir},trust_remote_code=True \
#     --gen_kwargs max_new_tokens=10 \
#     --tasks agieval_en \
#     --batch_size 8 \
#     --num_fewshot 3 \
#     --output_path ${output_dir}/eval_sft_agieval_en \
#     --seed 42 \

# accelerate launch -m lm_eval \
#     --model hf \
#     --model_args pretrained=${your_model_path},peft=${output_dir},trust_remote_code=True \
#     --gen_kwargs max_new_tokens=32 \
#     --tasks squadv2 \
#     --batch_size 8 \
#     --num_fewshot 1 \
#     --output_path ${output_dir}/eval_sft_squadv2 \
#     --seed 42 \

sleep 3

cd LLaMA-Factory
# # Eval mmlu
# llamafactory-cli \
#     eval ../config/${dataset}/${model_name}/eval_sft_mmlu.yaml \
#     adapter_name_or_path=${output_dir} \
#     finetuning_type=lora \
#     save_dir=${output_dir}/eval_sft_mmlu

sleep 3

# # Eval ceval
# llamafactory-cli \
#     eval ../config/${dataset}/${model_name}/eval_sft_ceval.yaml \
#     adapter_name_or_path=${output_dir} \
#     finetuning_type=lora \
#     save_dir=${output_dir}/eval_sft_ceval

sleep 3
# done

# Eval cmmlu
# llamafactory-cli \
#     eval ../config/${dataset}/${model_name}/eval_sft_cmmlu.yaml \
#     adapter_name_or_path=${output_dir} \
#     finetuning_type=lora \
#     save_dir=${output_dir}/eval_sft_cmmlu
# cd ..