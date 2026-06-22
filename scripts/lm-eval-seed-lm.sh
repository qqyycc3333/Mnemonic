#!/bin/bash
# chmod -R 777 ./LLaMA-Factory
chmod -R 777 ./lm-evaluation-harness

# 1. load model
your_model_path="./model/llama3-8b"
python ./model/download.py

echo "Download Model DONE"
# pwd: /checkpoint/binary/train_package

# 2. load lm-evaluation-harness llama-factory evaluate
cd ./lm-evaluation-harness
pip install -e .
cd ..

cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ..

cd ./evaluate
pip install -e .
cd ..

echo "Load lm-evaluation-harness & llama-factory & evaluate DONE"

# 3. register env path
add_path=/root/.local/bin
export PATH=$PATH:$add_path
echo $(which accelerate)

# 4. experiments
echo "begin lm-eval"

# output_path="/data/oss_bucket_0/yuanyi/aaawin/saves/llama3-8b/eval_vanilla_mmlu"

# accelerate launch -m lm_eval \
#     --model hf \
#     --model_args pretrained=${your_model_path},trust_remote_code=True \
#     --gen_kwargs max_new_tokens=3 \
#     --tasks mmlu \
#     --batch_size 8 \
#     --num_fewshot 5 \
#     --output_path ${output_path} \
#     --seed 42

# echo "mmlu DONE"

# output_path="/data/oss_bucket_0/yuanyi/aaawin/saves/llama3-8b/eval_vanilla_arc_challenge"

# accelerate launch -m lm_eval \
#     --model hf \
#     --model_args pretrained=${your_model_path},trust_remote_code=True \
#     --gen_kwargs max_new_tokens=3 \
#     --tasks arc_challenge \
#     --batch_size 8 \
#     --num_fewshot 25 \
#     --output_path ${output_path} \
#     --seed 42

# echo "arc_challenge DONE"


# output_path="/data/oss_bucket_0/yuanyi/aaawin/saves/llama3-8b/eval_vanilla_agieval_en"

# accelerate launch -m lm_eval \
#     --model hf \
#     --model_args pretrained=${your_model_path},trust_remote_code=True \
#     --gen_kwargs max_new_tokens=10 \
#     --tasks agieval_en \
#     --batch_size 8 \
#     --num_fewshot 3 \
#     --output_path ${output_path} \
#     --seed 42 \

# echo "agieval_en DONE"

# output_path="/data/oss_bucket_0/yuanyi/aaawin/saves/llama3-8b/eval_vanilla_squadv2"

# accelerate launch -m lm_eval \
#     --model hf \
#     --model_args pretrained=${your_model_path},trust_remote_code=True \
#     --gen_kwargs max_new_tokens=32 \
#     --tasks squadv2 \
#     --batch_size 8 \
#     --num_fewshot 1 \
#     --output_path ${output_path} \
#     --seed 42 \

# echo "squdv2 DONE"


# output_path="/data/oss_bucket_0/yuanyi/aaawin/saves/llama3-8b/eval_vanilla_gsm8k"

# accelerate launch -m lm_eval \
#     --model hf \
#     --model_args pretrained=${your_model_path},peft=${output_dir},trust_remote_code=True \
#     --gen_kwargs max_new_tokens=128 \
#     --tasks gsm8k \
#     --batch_size 8 \
#     --num_fewshot 0 \
#     --output_path ${output_path}/eval_sft_gsm8k \
#     --seed 42 

# echo "gsm8k DONE" 

# output_path="/data/oss_bucket_0/yuanyi/aaawin/saves/llama3-8b/eval_vanilla_drop"

# accelerate launch -m lm_eval \
#     --model hf \
#     --model_args pretrained=${your_model_path},trust_remote_code=True \
#     --gen_kwargs max_new_tokens=32 \
#     --tasks drop \
#     --num_fewshot 3 \
#     --batch_size 8 \
#     --output_path ${output_path} \
#     --seed 42 \

# echo "drop DONE"

# cd LLaMA-Factory
# # Eval ceval
# llamafactory-cli \
#     eval ../config/seed-lm/llama3-8b/eval_ceval.yaml \
    # adapter_name_or_path=${output_path} \
    # finetuning_type=full

# sleep 10

# # Eval cmmlu
# llamafactory-cli \
#     eval ../config/seed-lm/llama3-8b/eval_cmmlu.yaml \
    # adapter_name_or_path=${output_path} \
    # finetuning_type=full

export HF_ALLOW_CODE_EVAL=1
output_path="/data/oss_bucket_0/yuanyi/aaawin/saves/llama3-8b/eval_vanilla_humaneval"

accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=${your_model_path},trust_remote_code=True \
    --gen_kwargs max_new_tokens=1024 \
    --tasks humaneval \
    --num_fewshot 0 \
    --batch_size 8 \
    --output_path ${output_path} \
    --seed 42 \
    --confirm_run_unsafe_code
    
echo "humaneval DONE"
    