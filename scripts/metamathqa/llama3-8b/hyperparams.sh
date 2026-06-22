set -e

chmod -R 777 ./LLaMA-Factory

output_dir=
# output_dir="./saves/llama3-8b/hyperparams"

cd LLaMA-Factory
for learning_rate in 1e-3 3e-3 5e-3 1e-4 5e-4 5e-5
do
    for lora_rank in 8 16 64 128 
    do
        llamafactory-cli train ../config/metamathqa/llama3-8b/sft.yaml \
            output_dir=${output_dir}/${learning_rate}_${lora_rank} \
            learning_rate=${learning_rate} \
            lora_rank=${lora_rank} \
            max_samples=2000
    done
done
cd ..
