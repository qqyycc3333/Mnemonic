#!/bin/bash
chmod -R 777 ./LLaMA-Factory
# chmod -R 777 ./lm-evaluation-harness
# chmod -R 777 ./evaluate

# 1. load model
your_model_path="./model/llama3-8b"
python ./model/download.py
# python ./model/download_qwen.py

echo "Download Model DONE"
# pwd: /checkpoint/binary/train_package

# 2. load lm-evaluation-harness llama-factory evaluate
cd ./lm-evaluation-harness
pip install -e .
cd ..

cd ./LLaMA-Factory
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
dataset="metamathqa"
echo "begin sft ${dataset}"

sh ./scripts/metamathqa/llama3-8b/hyperparams.sh

