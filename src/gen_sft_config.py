import os

import pathlib

# Constants
# ENV_PATH = "~/envs/aaawin/bin"    # modify
PATH = "./LLaMA-Factory"    # modify
SAVE = "/data/oss_bucket_0/yuanyi/aaawin/saves"
# Configurations
# _cuda_visible_devices="0"

_model_name="llama3-8b"
_stage="sft"
_template="llama3_metamathqa"

_dataset="metamathqa"
_train_dataset=rf"{_dataset}_train"
# TODO
_eval_dataset="gsm8k_eval"

_num_train_epochs=1
_learning_rate=5.0e-4
_max_samples=50000
_max_new_tokens=128
_logging_steps=10
_save_steps=50
_eval_steps=50
_lora_rank=256
_cutoff_len=4096

# Path
_model_path=rf"../model/{_model_name}"
_output_dir=rf"{SAVE}/{_dataset}/{_model_name}/{_stage}"
_eval_dir=rf"{_output_dir}"

# Experiment name
_run_name = rf"./scripts/{_dataset}/{_model_name}"
config_path = rf"../config/{_dataset}/{_model_name}"

pathlib.Path(_run_name).mkdir(parents=True, exist_ok=True)
pathlib.Path(config_path).mkdir(parents=True, exist_ok=True)


# sft config
content_sft = rf"""
### model
model_name_or_path: {_model_path}
trust_remote_code: true

### method
stage: {_stage}                                         
finetuning_type: lora
lora_rank: {_lora_rank}
lora_target: q_proj,v_proj

### dataset
dataset: {_train_dataset}                                         
template: {_template}                         
cutoff_len: {_cutoff_len}
max_samples: {_max_samples}
overwrite_cache: true

### output
output_dir: {_output_dir}     
logging_steps: {_logging_steps}
save_steps: {_save_steps}
plot_loss: true
overwrite_output_dir: true
save_only_model: false

### train
do_train: true
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
num_train_epochs: {_num_train_epochs}
learning_rate: {_learning_rate}
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
# load_best_model_at_end: true

### eval
# do_eval: true
# eval_dataset: {_eval_dataset}  
# val_size: 0.1
# per_device_eval_batch_size: 4
# eval_strategy: steps
# eval_steps: {_eval_steps}
"""

with open(os.path.join(config_path, "sft.yaml"), 'w') as f:
    f.write(content_sft)
    

# predict config
content_predict = rf"""
### model
model_name_or_path: {_model_path}
adapter_name_or_path: {_output_dir}

### method
stage: {_stage}
do_predict: true
finetuning_type: lora

### dataset
eval_dataset: {_eval_dataset}
template: {_template}
cutoff_len: {_cutoff_len}
max_samples: {_max_samples}
overwrite_cache: true

### output
output_dir: {_output_dir}
overwrite_output_dir: false

### predict
per_device_eval_batch_size: 4
predict_with_generate: true
ddp_timeout: 180000000
max_new_tokens: {_max_new_tokens}

### hyperparameters
# gready search
# do_sample: true
# temperature: 0.95
# repetition_penalty: 1
# num_beams: 1
"""

with open(os.path.join(config_path, "predict.yaml"), 'w') as f:
    f.write(content_predict)


# scripts
content_sh = rf"""
#!/bin/bash
set -e
# source scripts_0/utils.sh
chmod -R 777 {PATH}

# Configurations
stage={_stage}

model_name={_model_name}
template={_template}

dataset={_dataset}
train_dataset={_train_dataset}
eval_dataset={_eval_dataset}

# Path
proj_path={PATH}
model_path={_model_path}
output_dir={_output_dir}
eval_dir={_eval_dir}

# Hyperparameters
learning_rate={_learning_rate}
lora_rank={_lora_rank}

# Debug
# cuda_visible_devices=0   
max_samples={_max_samples}
# max_new_tokens={_max_new_tokens}
""" + r"""
# create_empty_file ${result_file}
# echo -e "Fine-tuning using {stage}, {dataset}, {model_name}\n" >> ${result_file}

# Training
llamafactory-cli \
    train ./config/${dataset}/${model_name}/sft.yaml \
    dataset=${dataset}_train \
    output_dir=${output_dir} \
    max_samples=${max_samples}

sleep 10

# Predict
llamafactory-cli \
    train ./config/${dataset}/${model_name}/predict.yaml \
    adapter_name_or_path=${output_dir} \
    eval_dataset=${eval_dataset} \
    output_dir=${output_dir} \
    max_samples=${max_samples}

sleep 10

# Eval mmlu
llamafactory-cli \
    eval ./config/${dataset}/${model_name}/eval_mmlu.yaml \
    adapter_name_or_path=${output_dir} \
    finetuning_type=lora

sleep 10

# Eval ceval
llamafactory-cli \
    eval ./config/${dataset}/${model_name}/eval_ceval.yaml \
    adapter_name_or_path=${output_dir} \
    finetuning_type=lora

sleep 10

# Eval cmmlu
llamafactory-cli \
    eval ./config/${dataset}/${model_name}/eval_cmmlu.yaml \
    adapter_name_or_path=${output_dir} \
    finetuning_type=lora
"""

with open(os.path.join(_run_name, "run_sft.sh"), 'w') as f:
    f.write(content_sh)


# eval config
for task, lang in zip(["mmlu_test", "ceval_validation", "cmmlu_test"], ["en", "zh", "zh"]):
    name = task.split("_")[0]
    content_eval = rf"""
### model
model_name_or_path: {_model_path}
trust_remote_code: true
adapter_name_or_path: {_output_dir}

### method
finetuning_type: lora

### dataset
task: {task}    # mmlu_test, ceval_validation, cmmlu_test
template: fewshot
lang: {lang}    # en, zh
n_shot: 5

### output
save_dir: {_eval_dir}/eval_sft_{name}

### eval
batch_size: 4
"""
    with open(os.path.join(config_path, f"eval_{name}.yaml"), 'w') as f:
        f.write(content_eval)


# hyperparameters sh
hyperparams_path = rf"{SAVE}/{_dataset}/{_model_name}/hyperparams"
content_hparams = rf"""
for learning_rate in 1e-4 5e-4 1e-3 5e-5
do
    for lora_rank in 8 64 256
    do
        llamafactory-cli train ./config/{_dataset}/{_model_name}/sft.yaml \
            output_dir={hyperparams_path}/${{learning_rate}}_${{lora_rank}} \
            learning_rate=${{learning_rate}} \
            lora_rank=${{lora_rank}} \
            max_samples=2000
    done
done
"""

with open(os.path.join(_run_name, "hyperparams.sh"), 'w') as f:
    f.write(content_hparams)

# predict sh
content_predict_ckpt_sh = rf"""
#!/bin/bash
set -e
# source scripts_0/utils.sh
chmod -R 777 {PATH}

# Configurations
model_name={_model_name}
stage={_stage}
template={_template}
dataset={_dataset}
train_dataset={_train_dataset}
eval_dataset={_eval_dataset}

# Path
proj_path={PATH}
model_path={_model_path}
output_dir_sft={_output_dir}
eval_dir={_eval_dir}

# Hyperparameters
learning_rate={_learning_rate}
lora_rank={_lora_rank}

# Debug
# cuda_visible_devices=0   
max_samples=50000
# max_new_tokens={_max_new_tokens}
""" + r"""
# create_empty_file ${result_file}
# echo -e "Fine-tuning using {stage}, {dataset}, {model_name}\n" >> ${result_file}

# Training
# llamafactory-cli \
#     train ${proj_path}/config/${dataset}/${model_name}/sft.yaml \
#     dataset=${dataset}_train \
#     output_dir=${output_dir} \
#     max_samples=${max_samples}

sleep 10

for ckpt in 200 400 600 1000 1200 1400 1600 1800
do
    ckpt=checkpoint-${ckpt}
    output_dir=${output_dir_sft}/${ckpt}
    # Predict
    llamafactory-cli \
        train ./config/${dataset}/${model_name}/predict.yaml \
        adapter_name_or_path=${output_dir} \
        eval_dataset=${eval_dataset} \
        output_dir=${output_dir} \
        max_samples=${max_samples}

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
done
"""

with open(os.path.join(_run_name, "run_predict_ckpt.sh"), 'w') as f:
    f.write(content_predict_ckpt_sh)